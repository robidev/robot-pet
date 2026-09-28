import asyncio
import json
import socket
import struct

import pytest

from petd.bus import EventBus
from petd.config import Config, ConfigError
from petd.events import Heard, HeardDropped, SpeechEnded, SpeechStarted
from petd.io.stt import LocalMic, SttAdapter, drop_reason, lga1_packet


def make(echo_spans=()):
    bus = EventBus()
    def gate(a, b):     # the share of [a, b] the spans cover
        if b <= a:
            return float(any(s <= a <= e for s, e in echo_spans))
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in echo_spans) / (b - a)
    return SttAdapter(Config(), bus, gate), bus


def line(**kw):
    return json.dumps(kw)


def test_parses_whisper_json_lines():
    stt, bus = make()
    sub = bus.subscribe()
    stt.handle_line(line(type="ready", port=5000))
    stt.handle_line(line(type="speech_start", t_utc=100.0))
    stt.handle_line(line(type="speech_end", t_utc=102.0, duration_s=2.0, discarded=False))
    stt.handle_line(line(type="text", text=" Hello GLaDOS. ", t_start_utc=100.0, t_end_utc=102.0,
                         no_speech_prob=0.01, transcribe_ms=900))
    stt.handle_line("not json at all")
    events = [sub.get_nowait() for _ in range(3)]
    assert isinstance(events[0], SpeechStarted) and events[0].t_utc == 100.0
    assert isinstance(events[1], SpeechEnded)
    assert isinstance(events[2], Heard) and events[2].text == "Hello GLaDOS."
    assert sub.get_nowait() is None


def test_echo_of_own_speech_is_dropped():
    stt, bus = make(echo_spans=[(99.0, 101.0)])
    sub = bus.subscribe()
    stt.handle_line(line(type="speech_start", t_utc=100.5))          # suppressed
    stt.handle_line(line(type="text", text="The cake is a lie", t_start_utc=99.5,
                         t_end_utc=101.2, no_speech_prob=0.0))
    stt.handle_line(line(type="text", text="Are you still there", t_start_utc=110.0,
                         t_end_utc=111.0, no_speech_prob=0.0))
    first, second = sub.get_nowait(), sub.get_nowait()
    assert isinstance(first, HeardDropped) and first.reason == "echo of own speech"
    assert isinstance(second, Heard) and second.text == "Are you still there"


def test_an_answer_begun_as_the_voice_dies_away_is_kept():
    # 11:40 on 2026-09-23: a whole question dropped for overlapping the tail.
    stt, bus = make(echo_spans=[(99.0, 101.0)])
    sub = bus.subscribe()
    stt.handle_line(line(type="text", text="Yes, move back to the dock first", t_start_utc=100.6,
                         t_end_utc=105.0, no_speech_prob=0.0))
    assert isinstance(sub.get_nowait(), Heard)


def test_hallucination_filters():
    cfg = Config().stt
    assert drop_reason("[BLANK_AUDIO]", 0, cfg) == "non-speech tag"
    assert drop_reason("(wind blowing)", 0, cfg) == "non-speech tag"
    assert drop_reason("Thank you.", 0, cfg) == "ignore list"
    assert drop_reason("Thank you for watching.", 0, cfg) == "ignore list"   # motor noise, 2026-09-23
    assert drop_reason("a", 0, cfg) == "too short"
    assert drop_reason("go to the kitchen", 0.9, cfg).startswith("no_speech_prob")
    assert drop_reason("go to the kitchen", 0.1, cfg) is None


def test_mic_packets_are_what_whisper_parses():
    pcm = bytes(range(256)) * 4                        # 512 samples
    packet = lga1_packet(7, pcm)
    # udp-stream.cpp parse_packet: a 20-byte little-endian header, then samples.
    assert packet[:4] == b"LGA1"
    assert (packet[4], packet[5]) == (1, 1)            # version, channels
    assert struct.unpack_from("<H", packet, 6)[0] == 512
    assert struct.unpack_from("<I", packet, 8)[0] == 7
    assert struct.unpack_from("<I", packet, 16)[0] == 16000
    assert packet[20:] == pcm


async def test_local_mic_sends_the_recording_to_whisper():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    mic = LocalMic("test", rx.getsockname()[1])
    mic.process.argv = ["head", "-c", "2048", "/dev/zero"]   # arecord's stand-in: 64 ms
    mic.process.restart = False
    await mic.start()
    try:
        packets = [await asyncio.to_thread(rx.recv, 4096) for _ in range(2)]
    finally:
        await mic.close()
        rx.close()
    assert [struct.unpack_from("<I", p, 8)[0] for p in packets] == [0, 1]
    assert all(len(p) == 20 + 1024 for p in packets)


def port_of(stt):
    argv = stt.process.argv
    return int(argv[argv.index("--port") + 1])


def test_whisper_listens_where_the_audio_comes_from():
    cfg = Config()
    face = SttAdapter(cfg, EventBus())
    assert face.mic is None and port_of(face) == cfg.face.audio_port
    cfg.stt.source = "local"
    local = SttAdapter(cfg, EventBus())
    assert port_of(local) == cfg.stt.local_port != cfg.face.audio_port
    assert local.mic.addr == ("127.0.0.1", cfg.stt.local_port)
    cfg.stt.source = "jabra"
    with pytest.raises(ConfigError, match="stt.source"):
        SttAdapter(cfg, EventBus())
