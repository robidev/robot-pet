"""
Text-to-speech out of the robot's own speaker.

    text -> piper.http_server (/synthesize, WAV) -> raw S16LE PCM
         -> TCP robot:6000 -> socat -> aplay -f S16_LE -r 22050 -c 1

- An utterance is a stream of sentences (the brain adds them while the LLM
  is still generating). One TCP connection per utterance.
- Synthesis runs one sentence ahead of playback so there's no gap between
  sentences.
- Audio is paced to real time plus `lead_s`. Anything already sent can't
  be recalled, so a small lead keeps interrupt() fast.
- Playback spans are recorded so the STT echo gate can tell whether a
  transcript overlaps the pet's own voice.
"""

from __future__ import annotations

import asyncio
import io
import itertools
import logging
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from typing import Awaitable, Callable, Optional

from ..bus import EventBus
from ..config import Config, SpeakerConfig
from ..events import SpeakingFinished, SpeakingStarted
from ..net import tcp_port_open
from ..procs import ManagedProcess

log = logging.getLogger(__name__)

Synthesizer = Callable[[str], Awaitable[bytes]]   # text -> WAV bytes

CHUNK_S = 0.05


def wav_to_pcm(wav_bytes: bytes, expected_rate: int) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"expected 16-bit mono WAV, got {w.getsampwidth() * 8}-bit x{w.getnchannels()}")
        if w.getframerate() != expected_rate:
            raise ValueError(f"WAV is {w.getframerate()} Hz but the robot plays {expected_rate} Hz")
        return w.readframes(w.getnframes())


class AudioSink(ABC):
    @abstractmethod
    async def open(self) -> None: ...
    @abstractmethod
    async def write(self, pcm: bytes) -> None: ...
    @abstractmethod
    async def close(self) -> None: ...


class RobotTcpSink(AudioSink):
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._writer: Optional[asyncio.StreamWriter] = None

    async def open(self) -> None:
        # socat forks an aplay per connection; right after a previous
        # utterance it can briefly refuse the next one.
        last: Exception | None = None
        for attempt in range(3):
            try:
                _, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), 3.0)
                return
            except (OSError, asyncio.TimeoutError) as exc:
                last = exc
                await asyncio.sleep(0.4 * (attempt + 1))
        raise ConnectionError(
            f"robot speaker at {self.host}:{self.port} not reachable "
            f"(is socat running on the robot?): {last}") from last

    async def write(self, pcm: bytes) -> None:
        self._writer.write(pcm)
        await self._writer.drain()

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except OSError:
                pass
            self._writer = None


class NullSink(AudioSink):
    """Discards audio (timing is still real): for --fake mode and tests."""

    def __init__(self):
        self.bytes_written = 0

    async def open(self) -> None: ...

    async def write(self, pcm: bytes) -> None:
        self.bytes_written += len(pcm)

    async def close(self) -> None: ...


class Utterance:
    """Sentences to speak as one continuous stream. add() then end()."""
    _ids = itertools.count(1)

    def __init__(self):
        self.id = next(self._ids)
        self.sentences: asyncio.Queue = asyncio.Queue()
        self.spoken: list[str] = []
        self.done = asyncio.Event()
        self.interrupted = False

    def add(self, sentence: str) -> None:
        if sentence.strip():
            self.sentences.put_nowait(sentence.strip())

    def end(self) -> None:
        self.sentences.put_nowait(None)

    async def wait(self) -> None:
        await self.done.wait()


def piper_synthesizer(url: str, startup_timeout_s: float = 30.0) -> Synthesizer:
    import requests
    session = requests.Session()

    def post(text: str) -> bytes:
        # petd starts piper itself, and the flask server needs a few
        # seconds; the first utterance can easily beat it.
        deadline = time.monotonic() + startup_timeout_s
        while True:
            try:
                resp = session.post(f"{url}/synthesize", json={"text": text}, timeout=30)
                break
            except requests.ConnectionError:
                if time.monotonic() > deadline:
                    raise
                log.info("waiting for piper at %s ...", url)
                time.sleep(0.5)
        resp.raise_for_status()
        return resp.content

    async def synthesize(text: str) -> bytes:
        return await asyncio.to_thread(post, text)

    return synthesize


class Speaker:
    def __init__(self, cfg: SpeakerConfig, bus: EventBus, synthesize: Synthesizer,
                 sink_factory: Callable[[], AudioSink]):
        self.cfg = cfg
        self.bus = bus
        self.synthesize = synthesize
        self.sink_factory = sink_factory
        self._queue: asyncio.Queue = asyncio.Queue()
        self._current: Optional[Utterance] = None
        self._current_task: Optional[asyncio.Task] = None
        self._worker: Optional[asyncio.Task] = None
        # (start, end) wall-clock spans during which our voice is audible.
        self._spans: deque = deque(maxlen=50)
        self._open_span_start: Optional[float] = None
        self._open_span_end = 0.0

    # --- public API -----------------------------------------------------------

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="speaker")

    async def close(self) -> None:
        self.interrupt()
        if self._worker:
            self._worker.cancel()

    def begin(self) -> Utterance:
        utt = Utterance()
        self._queue.put_nowait(utt)
        return utt

    def say(self, text: str) -> Utterance:
        utt = self.begin()
        utt.add(text)
        utt.end()
        return utt

    def interrupt(self) -> None:
        """Stops the current utterance and drops everything queued."""
        while not self._queue.empty():
            utt = self._queue.get_nowait()
            utt.interrupted = True
            utt.done.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    @property
    def speaking(self) -> bool:
        return self.is_audible(time.time())

    def is_audible(self, t: float) -> bool:
        return self.overlaps(t, t)

    def overlaps(self, t_start: float, t_end: float) -> bool:
        """True if [t_start, t_end] overlaps our own audible speech (+ echo tail)."""
        tail = self.cfg.gate_tail_s
        spans = list(self._spans)
        if self._open_span_start is not None:
            spans.append((self._open_span_start, max(self._open_span_end, time.time())))
        return any(t_start <= end + tail and t_end >= start for start, end in spans)

    # --- internals ------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            utt: Utterance = await self._queue.get()
            if utt.done.is_set():       # interrupted while queued
                continue
            self._current = utt
            self._current_task = asyncio.create_task(self._speak(utt))
            try:
                await self._current_task
            except asyncio.CancelledError:
                utt.interrupted = True
                # interrupt() cancels only the utterance task; if the worker
                # itself is being cancelled (shutdown), propagate.
                if asyncio.current_task().cancelling():
                    raise
            except Exception:  # noqa: BLE001 - one bad utterance mustn't kill the speaker
                log.exception("utterance %d failed", utt.id)
            finally:
                self._close_span(interrupted=utt.interrupted)
                self._current = None
                utt.done.set()
                self.bus.publish(SpeakingFinished(utterance_id=utt.id, text=" ".join(utt.spoken),
                                                  interrupted=utt.interrupted))

    async def _speak(self, utt: Utterance) -> None:
        pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=1)   # one sentence of look-ahead
        synth = asyncio.create_task(self._synthesize_all(utt, pcm_queue))
        sink: Optional[AudioSink] = None
        try:
            bytes_per_s = self.cfg.sample_rate * 2
            chunk = int(CHUNK_S * bytes_per_s) & ~1
            t0 = 0.0
            sent_s = 0.0
            while True:
                item = await pcm_queue.get()
                if item is None:
                    break
                sentence, pcm = item
                if sink is None:
                    sink = self.sink_factory()
                    await sink.open()
                    t0 = time.monotonic()
                    self._open_span_start = time.time() + self.cfg.playback_latency_s
                    self._open_span_end = self._open_span_start
                    self.bus.publish(SpeakingStarted(utterance_id=utt.id))
                # If synthesis fell behind, playback has drained: restart the clock.
                now_s = time.monotonic() - t0
                if sent_s < now_s:
                    sent_s = now_s
                utt.spoken.append(sentence)
                for i in range(0, len(pcm), chunk):
                    ahead = sent_s - (time.monotonic() - t0)
                    if ahead > self.cfg.lead_s:
                        await asyncio.sleep(ahead - self.cfg.lead_s)
                    piece = pcm[i:i + chunk]
                    await sink.write(piece)
                    sent_s += len(piece) / bytes_per_s
                    self._open_span_end = time.time() + (sent_s - (time.monotonic() - t0)) \
                        + self.cfg.playback_latency_s
            # Let the tail of the audio actually play before declaring done.
            remaining = sent_s - (time.monotonic() - t0) + self.cfg.playback_latency_s
            if sink is not None and remaining > 0:
                await asyncio.sleep(remaining)
        finally:
            synth.cancel()
            if sink is not None:
                await sink.close()

    async def _synthesize_all(self, utt: Utterance, out: asyncio.Queue) -> None:
        while True:
            sentence = await utt.sentences.get()
            if sentence is None:
                await out.put(None)
                return
            try:
                pcm = wav_to_pcm(await self.synthesize(sentence), self.cfg.sample_rate)
            except Exception:  # noqa: BLE001 - skip the sentence, keep talking
                log.exception("synthesis failed for %r", sentence)
                continue
            await out.put((sentence, pcm))

    def _close_span(self, interrupted: bool) -> None:
        if self._open_span_start is None:
            return
        end = self._open_span_end
        if interrupted:
            # Audio already pushed (lead + aplay buffer) keeps playing briefly.
            end = min(end, time.time() + self.cfg.lead_s + self.cfg.playback_latency_s)
        self._spans.append((self._open_span_start, end))
        self._open_span_start = None


def build_speaker(cfg: Config, bus: EventBus, fake: bool) -> tuple[Speaker, Optional[ManagedProcess]]:
    """The Speaker plus, if configured and not already running, a managed piper server."""
    sc = cfg.speaker
    if fake or sc.sink == "null":
        sink_factory: Callable[[], AudioSink] = NullSink
    else:
        sink_factory = lambda: RobotTcpSink(cfg.vacuum.host, sc.robot_port)  # noqa: E731

    if fake:
        async def synthesize(text: str) -> bytes:
            # Silence roughly as long as the text would take to say.
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sc.sample_rate)
                w.writeframes(b"\x00\x00" * int(sc.sample_rate * 0.06 * len(text)))
            return buf.getvalue()
        return Speaker(sc, bus, synthesize, sink_factory), None

    piper = None
    from urllib.parse import urlparse
    u = urlparse(sc.piper_url)
    if sc.manage_piper and not tcp_port_open(u.hostname, u.port or 80):
        piper = ManagedProcess("piper", sc.piper_cmd, cwd=cfg.path(sc.piper_cwd), bus=bus)
    return Speaker(sc, bus, piper_synthesizer(sc.piper_url), sink_factory), piper
