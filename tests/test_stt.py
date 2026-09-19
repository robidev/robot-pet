import json

from petd.bus import EventBus
from petd.config import Config
from petd.events import Heard, HeardDropped, SpeechEnded, SpeechStarted
from petd.io.stt import SttAdapter, drop_reason


def make(echo_spans=()):
    bus = EventBus()
    gate = lambda a, b: any(a <= e and b >= s for s, e in echo_spans)  # noqa: E731
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
    stt.handle_line(line(type="text", text="The cake is a lie", t_start_utc=100.5,
                         t_end_utc=103.0, no_speech_prob=0.0))
    stt.handle_line(line(type="text", text="Are you still there", t_start_utc=110.0,
                         t_end_utc=111.0, no_speech_prob=0.0))
    first, second = sub.get_nowait(), sub.get_nowait()
    assert isinstance(first, HeardDropped) and first.reason == "echo of own speech"
    assert isinstance(second, Heard) and second.text == "Are you still there"


def test_hallucination_filters():
    cfg = Config().stt
    assert drop_reason("[BLANK_AUDIO]", 0, cfg) == "non-speech tag"
    assert drop_reason("(wind blowing)", 0, cfg) == "non-speech tag"
    assert drop_reason("Thank you.", 0, cfg) == "ignore list"
    assert drop_reason("a", 0, cfg) == "too short"
    assert drop_reason("go to the kitchen", 0.9, cfg).startswith("no_speech_prob")
    assert drop_reason("go to the kitchen", 0.1, cfg) is None
