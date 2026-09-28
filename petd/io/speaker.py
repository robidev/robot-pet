"""
Text-to-speech out of the robot's own speaker, or a sound card on this PC.

    text -> piper.http_server (/synthesize, WAV) -> raw S16LE PCM
         -> TCP robot:6000 -> socat -> aplay -f S16_LE -r 22050 -c 1
            (speaker.sink: robot)
         -> aplay -D speaker.local_device on this PC (speaker.sink: local)

- An utterance is a stream of sentences (the brain adds them while the LLM
  is still generating). One TCP connection per utterance.
- Synthesis runs one sentence ahead of playback so there's no gap between
  sentences.
- Audio is paced to real time plus `lead_s`, and the lead is the jitter
  buffer: the robot's WiFi stalls for over a second now and then (pings
  of 1.3-1.4 s, while the face and router on the same network stay under
  20 ms), and with a 0.3 s lead every stall was an aplay underrun, heard
  as the voice cutting out. What's already sent can't be recalled, so
  interrupt() resets the connection and, if the robot has one, pokes its
  stop port, which kills aplay and the queued audio with it.
- Playback spans are recorded so the STT echo gate can tell whether a
  transcript overlaps the pet's own voice.
- The robot has no usable mixer (amixer controls nothing on its sound
  card) and piper's HTTP server doesn't expose piper's own --volume, so
  loudness is set here, by scaling the PCM before it goes out.
"""

from __future__ import annotations

import array
import asyncio
import io
import itertools
import logging
import sys
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from typing import Awaitable, Callable, Optional

from ..bus import EventBus
from ..config import Config, ConfigError, SpeakerConfig
from ..events import SentenceSynthesized, SpeakingFinished, SpeakingStarted
from ..net import tcp_port_open
from ..procs import ManagedProcess

log = logging.getLogger(__name__)

Synthesizer = Callable[[str], Awaitable[bytes]]   # text -> WAV bytes

CHUNK_S = 0.05
# How long the robot keeps playing after interrupt() pokes its stop port.
INTERRUPT_S = 0.5


def wav_to_pcm(wav_bytes: bytes, expected_rate: int) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"expected 16-bit mono WAV, got {w.getsampwidth() * 8}-bit x{w.getnchannels()}")
        if w.getframerate() != expected_rate:
            raise ValueError(f"WAV is {w.getframerate()} Hz but the robot plays {expected_rate} Hz")
        return w.readframes(w.getnframes())


def apply_gain(pcm: bytes, gain: float) -> bytes:
    """Scales S16LE samples by `gain`, clipping if asked for more than 1.0."""
    if gain == 1.0:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    for i, s in enumerate(samples):
        v = int(s * gain)
        samples[i] = -32768 if v < -32768 else 32767 if v > 32767 else v
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


class AudioSink(ABC):
    @abstractmethod
    async def open(self) -> None: ...
    @abstractmethod
    async def write(self, pcm: bytes) -> None: ...
    @abstractmethod
    async def close(self) -> None: ...

    async def abort(self) -> None:
        """Stops at once, dropping whatever is queued downstream."""
        await self.close()


class RobotTcpSink(AudioSink):
    def __init__(self, host: str, port: int, stop_port: Optional[int] = None):
        self.host, self.port, self.stop_port = host, port, stop_port
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

    async def abort(self) -> None:
        # A reset, not a FIN. On its own that still takes ~2 s to go quiet:
        # socat only notices once aplay has drained the pipe in between, and
        # then gives aplay a second's grace. A connection to the robot's stop
        # port (a socat that runs `killall aplay`) silences it at once.
        if self._writer is not None:
            self._writer.transport.abort()
            self._writer = None
        if self.stop_port:
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.stop_port), 1.0)
                writer.close()
            except (OSError, asyncio.TimeoutError) as exc:
                log.warning("robot stop port %s:%s: %s", self.host, self.stop_port, exc)


class LocalSink(AudioSink):
    """aplay on this PC, one per utterance as on the robot; abort() kills it."""

    def __init__(self, device: str, sample_rate: int):
        self.device = device
        self.argv = ["aplay", "-q", "-D", device, "-t", "raw",
                     "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"]
        self._proc: Optional[asyncio.subprocess.Process] = None

    async def open(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.argv, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        except OSError as exc:
            raise ConnectionError(f"can't start aplay: {exc}") from exc

    async def write(self, pcm: bytes) -> None:
        try:
            self._proc.stdin.write(pcm)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            # aplay gave up: no such device, busy, or no permission.
            err = (await self._proc.stderr.read()).decode(errors="replace").strip()
            await self.abort()
            raise ConnectionError(f"aplay on {self.device!r} stopped: {err or exc}") from exc

    async def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        proc.stdin.close()                  # aplay plays what it has, then exits
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def abort(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


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
        # (time sent, sentence): what the mic may be hearing of us right now.
        self._said: deque = deque(maxlen=20)

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

    def said_recently(self, window_s: float = 30.0) -> str:
        """Our own sentences sent in the last window_s, for telling echo from barge-in."""
        cutoff = time.time() - window_s
        return " ".join(sentence for t, sentence in self._said if t >= cutoff)

    @property
    def speaking(self) -> bool:
        return self.is_audible(time.time())

    def is_audible(self, t: float) -> bool:
        return self.overlaps(t, t)

    def overlap_fraction(self, t_start: float, t_end: float) -> float:
        """How much of [t_start, t_end] our own audible speech (+ echo tail) covers, 0..1."""
        if t_end <= t_start:
            return 1.0 if self.overlaps(t_start, t_start) else 0.0
        tail = self.cfg.gate_tail_s
        spans = list(self._spans)
        if self._open_span_start is not None:
            spans.append((self._open_span_start, max(self._open_span_end, time.time())))
        covered, cursor = 0.0, t_start
        for start, end in sorted((s, e + tail) for s, e in spans):
            start, end = max(start, cursor), min(end, t_end)
            if end > start:
                covered += end - start
                cursor = end
        return covered / (t_end - t_start)

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
                self._said.append((time.time(), sentence))
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
        except asyncio.CancelledError:
            if sink is not None:
                await sink.abort()
                sink = None
            raise
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
            started = time.monotonic()
            try:
                pcm = wav_to_pcm(await self.synthesize(sentence), self.cfg.sample_rate)
                pcm = apply_gain(pcm, self.cfg.volume)
            except Exception:  # noqa: BLE001 - skip the sentence, keep talking
                log.exception("synthesis failed for %r", sentence)
                continue
            self.bus.publish(SentenceSynthesized(
                utterance_id=utt.id, text=sentence, synth_s=time.monotonic() - started,
                audio_s=len(pcm) / (2 * self.cfg.sample_rate)))
            await out.put((sentence, pcm))

    def _close_span(self, interrupted: bool) -> None:
        if self._open_span_start is None:
            return
        end = self._open_span_end
        if interrupted:
            # With a stop port the robot goes quiet almost at once, as a
            # local sink does. Without one, it plays until socat sees the
            # reset (after the lead has drained through the pipe) and its
            # 1 s grace for aplay is up.
            instant = self.cfg.stop_port or self.cfg.sink != "robot"
            quiet_after = INTERRUPT_S if instant else self.cfg.lead_s + 1.0
            end = min(end, time.time() + quiet_after + self.cfg.playback_latency_s)
        self._spans.append((self._open_span_start, end))
        self._open_span_start = None


def build_speaker(cfg: Config, bus: EventBus, fake: bool) -> tuple[Speaker, Optional[ManagedProcess]]:
    """The Speaker plus, if configured and not already running, a managed piper server."""
    sc = cfg.speaker
    if sc.sink not in ("robot", "local", "null"):
        raise ConfigError(f"speaker.sink must be robot, local or null, not {sc.sink!r}")
    if fake or sc.sink == "null":
        sink_factory: Callable[[], AudioSink] = NullSink
    elif sc.sink == "local":
        sink_factory = lambda: LocalSink(sc.local_device, sc.sample_rate)  # noqa: E731
    else:
        sink_factory = lambda: RobotTcpSink(cfg.vacuum.host, sc.robot_port, sc.stop_port)  # noqa: E731

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
