"""
Speech-to-text: supervises whisper-udp-stream in --json mode and turns its
output into bus events.

Filtering happens here, before anything reaches the brain:
- echo: the utterance overlaps the pet's own playback (the mic is on the
  same body as the speaker), judged by the utterance's own timestamps since
  transcription lands ~1-2 s after the speech ended;
- Whisper hallucinations on noise: bracketed tags ("[BLANK_AUDIO]"),
  a configurable ignore list, too-short text, high no_speech_prob.
Dropped transcripts are published as HeardDropped for debugging.
"""

from __future__ import annotations

import json
import logging
import re
import string
import time
from typing import Callable, Optional

from ..bus import EventBus
from ..config import Config, SttConfig
from ..events import Heard, HeardDropped, SpeechEnded, SpeechStarted
from ..procs import ManagedProcess

log = logging.getLogger(__name__)

# (t_start, t_end) -> True if that span overlaps the pet's own speech.
EchoGate = Callable[[float, float], bool]

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


class SttAdapter:
    def __init__(self, cfg: Config, bus: EventBus, echo_gate: Optional[EchoGate] = None):
        self.cfg = cfg.stt
        self.bus = bus
        self.echo_gate = echo_gate
        c = cfg.stt
        argv = [
            str(cfg.path(c.binary)), "--json",
            "--port", str(cfg.face.audio_port),
            "--threads", str(c.threads),
            "--model", c.model, "--vad-model", c.vad_model,
            "--prompt", c.prompt if c.prompt is not None else cfg.pet.name,
            *[str(a) for a in c.extra_args],
        ]
        self.process = ManagedProcess(
            "stt", argv, cwd=cfg.path(c.cwd), on_stdout_line=self.handle_line, bus=bus)

    async def start(self) -> None:
        self.process.start()

    async def close(self) -> None:
        await self.process.stop()

    def _echo(self, t_start: float, t_end: float) -> bool:
        return self.echo_gate is not None and self.echo_gate(t_start, t_end)

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
            if not self._echo(t, t):
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
        reason = "echo of own speech" if self._echo(t_start, t_end) else drop_reason(text, no_speech_prob, self.cfg)
        if reason:
            log.debug("dropped %r: %s", text, reason)
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
