import asyncio
import threading

import pytest

from petd.bus import EventBus
from petd.config import ConfigError, PROJECT_ROOT, load_config
from petd.events import Event, Heard, SpeechStarted


def test_defaults_and_example_config_load():
    cfg = load_config(PROJECT_ROOT / "config.example.yaml", environ={})
    assert cfg.pet.name == "GLaDOS"
    assert cfg.face.host == "192.168.101.40"
    assert cfg.path("stt/udp-stream").is_absolute()


def test_yaml_and_env_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("face:\n  host: 10.0.0.2\nstt:\n  threads: 2\n")
    cfg = load_config(f, environ={"PETD__STT__THREADS": "4", "PETD__FACE__AUDIO_GAIN": "1.5"})
    assert cfg.face.host == "10.0.0.2"
    assert cfg.stt.threads == 4
    assert cfg.face.audio_gain == 1.5


def test_unknown_key_is_an_error(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("face:\n  hots: 10.0.0.2\n")
    with pytest.raises(ConfigError, match="face.hots"):
        load_config(f, environ={})


async def test_bus_filters_by_type_and_keeps_history():
    bus = EventBus()
    heard = bus.subscribe(Heard)
    everything = bus.subscribe()
    bus.publish(SpeechStarted(t_utc=1.0))
    bus.publish(Heard(text="hi", t_start=1, t_end=2))
    assert (await heard.get()).text == "hi"
    assert heard.get_nowait() is None
    assert isinstance(await everything.get(), SpeechStarted)
    assert len(bus.history) == 2


async def test_bus_drops_oldest_when_full():
    bus = EventBus()
    sub = bus.subscribe(Event, maxsize=2)
    for i in range(3):
        bus.publish(SpeechStarted(t_utc=float(i)))
    assert [(await sub.get()).t_utc for _ in range(2)] == [1.0, 2.0]
    assert sub.dropped == 1


async def test_publish_threadsafe():
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    sub = bus.subscribe(SpeechStarted)
    threading.Thread(target=lambda: bus.publish_threadsafe(SpeechStarted(t_utc=5.0))).start()
    assert (await asyncio.wait_for(sub.get(), 2)).t_utc == 5.0
