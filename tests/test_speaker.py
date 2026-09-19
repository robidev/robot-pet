import asyncio
import io
import time
import wave

import pytest

from petd.bus import EventBus
from petd.config import SpeakerConfig
from petd.events import SpeakingFinished, SpeakingStarted
from petd.io.speaker import NullSink, Speaker, wav_to_pcm

RATE = 22050


def wav(seconds: float, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return buf.getvalue()


def make(seconds_per_sentence=0.3, **cfg):
    sink = NullSink()
    bus = EventBus()

    async def synth(text):
        return wav(seconds_per_sentence)

    c = SpeakerConfig(sink="null", playback_latency_s=0.0, lead_s=0.1, gate_tail_s=0.2, **cfg)
    return Speaker(c, bus, synth, lambda: sink), sink, bus


def test_wav_to_pcm_checks_rate():
    assert len(wav_to_pcm(wav(0.1), RATE)) == int(RATE * 0.1) * 2
    with pytest.raises(ValueError, match="16000 Hz"):
        wav_to_pcm(wav(0.1, 16000), RATE)


async def test_streams_all_sentences_paced_to_real_time():
    speaker, sink, bus = make()
    sub = bus.subscribe(SpeakingStarted, SpeakingFinished)
    await speaker.start()
    t0 = time.monotonic()
    utt = speaker.begin()
    utt.add("Hello.")
    utt.add("Welcome to the test.")
    utt.end()
    await asyncio.wait_for(utt.wait(), 5)
    elapsed = time.monotonic() - t0
    assert sink.bytes_written == 2 * int(RATE * 0.3) * 2
    assert 0.55 < elapsed < 1.5          # ~0.6 s of audio, paced, not dumped instantly
    assert isinstance(await sub.get(), SpeakingStarted)
    done = await sub.get()
    assert done.text == "Hello. Welcome to the test." and not done.interrupted
    await speaker.close()


async def test_interrupt_stops_quickly_and_drops_queue():
    speaker, sink, bus = make(seconds_per_sentence=3.0)
    await speaker.start()
    first = speaker.say("A very long monologue.")
    second = speaker.say("Never said.")
    await asyncio.sleep(0.3)
    t0 = time.monotonic()
    speaker.interrupt()
    await asyncio.wait_for(asyncio.gather(first.wait(), second.wait()), 1)
    assert time.monotonic() - t0 < 0.5
    assert first.interrupted and second.interrupted
    assert sink.bytes_written < RATE * 2 * 1.0     # well under the 3 s of audio
    await speaker.close()


async def test_echo_gate_spans():
    speaker, sink, bus = make(seconds_per_sentence=0.4)
    await speaker.start()
    before = time.time()
    utt = speaker.say("Testing.")
    await asyncio.sleep(0.2)
    assert speaker.speaking
    await utt.wait()
    after = time.time()
    assert speaker.overlaps(before + 0.1, before + 0.2)
    assert speaker.overlaps(after + 0.1, after + 0.1)          # within gate tail
    assert not speaker.overlaps(after + 1.0, after + 2.0)
    await speaker.close()
