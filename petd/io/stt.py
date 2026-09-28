"""
Speech-to-text: supervises whisper-udp-stream in --json mode and turns its
output into bus events.

Filtering happens here, before anything reaches the brain:
- echo (unless stt.echo_cancelled: the mic doesn't hear the pet at all): most of the utterance overlaps the pet's own playback (the mic is on
  the same body as the speaker), judged by the utterance's own timestamps
  since transcription lands ~1-2 s after the speech ended. Only most: an
  answer begun just before the pet's voice had quite died away is still
  someone talking (behavior/converse.py catches echoes by their words);
- Whisper hallucinations on noise: bracketed tags ("[BLANK_AUDIO]"),
  a configurable ignore list, too-short text, high no_speech_prob.
Dropped transcripts are published as HeardDropped for debugging.

The audio comes from the face's UDP stream (stt.source: face) or from a sound
card on this PC (stt.source: local): LocalMic records it with arecord and
sends it to whisper on localhost in the face's own packet format, so the
recognizer is the same either way.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import string
import struct
import time
from typing import Callable, Optional

from ..bus import EventBus
from ..config import Config, ConfigError, SttConfig
from ..events import Heard, HeardDropped, SpeechEnded, SpeechStarted
from ..procs import ManagedProcess

log = logging.getLogger(__name__)

# (t_start, t_end) -> how much of that span the pet's own speech covers, 0..1.
EchoGate = Callable[[float, float], float]

# whisper-udp-stream's packet (udp-stream.cpp): magic, version, channels,
# sample count, sequence, timestamp, sample rate, then S16LE samples.
_LGA1 = struct.Struct("<4sBBHIII")
MIC_RATE = 16000
MIC_PACKET_SAMPLES = 512                    # 32 ms, one VAD window

_BRACKETED = re.compile(r"^\s*[\[\(\*].*[\]\)\*]\s*$")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


def drop_reason(text: str, no_speech_prob: float, cfg: SttConfig) -> Optional[str]:
    """Why a transcript should be ignored, or None to keep it."""
    if _BRACKETED.match(text):
        return "non-speech tag"
    norm = normalize(text)
    if len(norm) < cfg.min_chars:
        return "too short"
    if norm in {normalize(p) for p in cfg.ignore_phrases}:
        return "ignore list"
    if no_speech_prob > cfg.max_no_speech_prob:
        return f"no_speech_prob {no_speech_prob:.2f}"
    return None


def lga1_packet(sequence: int, pcm: bytes) -> bytes:
    """16 kHz mono S16LE samples as one packet of the face's audio stream."""
    header = _LGA1.pack(b"LGA1", 1, 1, len(pcm) // 2, sequence & 0xFFFFFFFF,
                        int(time.monotonic() * 1000) & 0xFFFFFFFF, MIC_RATE)
    return header + pcm


class LocalMic:
    """A sound card on this PC in the face's place: arecord -> packets -> whisper on localhost."""

    def __init__(self, device: str, port: int, bus: Optional[EventBus] = None):
        self.addr = ("127.0.0.1", port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = 0
        argv = ["arecord", "-q", "-D", device, "-t", "raw",
                "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1"]
        # arecord exits when the card goes (unplugged, usbipd detached);
        # the supervisor keeps trying until it's back.
        self.process = ManagedProcess(
            "mic", argv, on_stdout_chunk=self.send, chunk_size=MIC_PACKET_SAMPLES * 2,
            stderr_level=logging.WARNING, bus=bus)

    def send(self, pcm: bytes) -> None:
        try:
            self._sock.sendto(lga1_packet(self._sequence, pcm), self.addr)
        except OSError as exc:
            log.debug("mic packet not sent: %s", exc)
        self._sequence += 1

    async def start(self) -> None:
        self.process.start()

    async def close(self) -> None:
        await self.process.stop()
        self._sock.close()


class SttAdapter:
    def __init__(self, cfg: Config, bus: EventBus, echo_gate: Optional[EchoGate] = None):
        self.cfg = cfg.stt
        self.bus = bus
        self.echo_gate = echo_gate
        c = cfg.stt
        if c.source not in ("face", "local"):
            raise ConfigError(f"stt.source must be face or local, not {c.source!r}")
        port = c.local_port if c.source == "local" else cfg.face.audio_port
        self.mic = LocalMic(c.local_device, port, bus) if c.source == "local" else None
        argv = [
            str(cfg.path(c.binary)), "--json",
            "--port", str(port),
            "--threads", str(c.threads),
            "--model", c.model, "--vad-model", c.vad_model,
            "--prompt", c.prompt if c.prompt is not None else cfg.pet.name,
            *[str(a) for a in c.extra_args],
        ]
        self.process = ManagedProcess(
            "stt", argv, cwd=cfg.path(c.cwd), on_stdout_line=self.handle_line, bus=bus)

    async def start(self) -> None:
        self.process.start()
        if self.mic is not None:
            await self.mic.start()

    async def close(self) -> None:
        if self.mic is not None:
            await self.mic.close()
        await self.process.stop()

    def _echo_share(self, t_start: float, t_end: float) -> float:
        if self.echo_gate is None or self.cfg.echo_cancelled:
            return 0.0
        return float(self.echo_gate(t_start, t_end))

    def handle_line(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            log.debug("non-JSON stt output: %s", line)
            return
        kind = msg.get("type")
        if kind == "speech_start":
            t = msg["t_utc"]
            # Our own voice starting up shouldn't make the pet "listen".
            if not self._echo_share(t, t):
                self.bus.publish(SpeechStarted(t_utc=t))
        elif kind == "speech_end":
            self.bus.publish(SpeechEnded(t_utc=msg["t_utc"], discarded=msg.get("discarded", False)))
        elif kind == "text":
            self.handle_text(msg.get("text", ""), msg["t_start_utc"], msg["t_end_utc"],
                             msg.get("no_speech_prob", 0.0))
        elif kind == "ready":
            log.info("stt ready on UDP port %s", msg.get("port"))

    def handle_text(self, text: str, t_start: float, t_end: float, no_speech_prob: float = 0.0,
                    source: str = "mic") -> None:
        text = text.strip()
        share = self._echo_share(t_start, t_end)
        reason = ("echo of own speech" if share >= self.cfg.echo_overlap
                  else drop_reason(text, no_speech_prob, self.cfg))
        if reason:
            # An echo drop may well be someone talking over the pet: worth seeing.
            level = logging.INFO if share >= self.cfg.echo_overlap else logging.DEBUG
            log.log(level, "dropped %r: %s (%.0f%% over my own voice)", text, reason, share * 100)
            self.bus.publish(HeardDropped(text=text, reason=reason))
            return
        log.info("heard: %r", text)
        self.bus.publish(Heard(text=text, t_start=t_start, t_end=t_end,
                               no_speech_prob=no_speech_prob, source=source))


class FakeStt:
    """No subprocess; inject() pushes text through the same filters."""

    def __init__(self, cfg: Config, bus: EventBus, echo_gate: Optional[EchoGate] = None):
        self._real = SttAdapter.__new__(SttAdapter)
        self._real.cfg, self._real.bus, self._real.echo_gate = cfg.stt, bus, echo_gate

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    def inject(self, text: str, source: str = "fake") -> None:
        now = time.time()
        self._real.handle_text(text, now - 1.0, now, source=source)
