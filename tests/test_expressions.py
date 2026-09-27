import asyncio
from types import SimpleNamespace

from petd.brain.brain import Brain
from petd.brain.expressions import Expressions
from petd.bus import EventBus
from petd.config import Config
from petd.events import HeardDropped, SpeechEnded, SpeechStarted
from petd.io.face import FakeFace


def eye_modes(face):
    return [c[1] for c in face.commands if c[0] == "eye_mode"]


def servo_modes(face):
    return [c[1] for c in face.commands if c[0] == "servo" and c[1] is not None]


async def feedback(bus, face):
    pet = SimpleNamespace(cfg=Config(), face=face, bus=bus)
    brain = Brain(pet, backend=None)
    task = asyncio.create_task(brain._feedback_loop(bus.subscribe(SpeechStarted, SpeechEnded, HeardDropped)))
    await asyncio.sleep(0)
    return brain, task


async def test_a_dropped_transcript_hands_the_eye_back():
    # 2026-09-27: [BLANK_AUDIO], dropped by the stt adapter, left the eye on "thinking".
    bus = EventBus()
    face = FakeFace(Config().face, bus)
    brain, task = await feedback(bus, face)
    bus.publish(SpeechStarted(t_utc=0.0))
    bus.publish(SpeechEnded(t_utc=1.0))
    bus.publish(HeardDropped(text="[BLANK_AUDIO]", reason="non-speech tag"))
    await asyncio.sleep(0.05)
    assert eye_modes(face)[-1] == "auto"

    brain.busy = True                       # mid-turn: the turn hands it back when it ends
    bus.publish(SpeechEnded(t_utc=2.0))
    bus.publish(HeardDropped(text="you", reason="echo of own speech"))
    await asyncio.sleep(0.05)
    assert eye_modes(face)[-1] == "manual"
    task.cancel()


async def test_thinking_with_nothing_after_it_times_out(monkeypatch):
    monkeypatch.setattr("petd.brain.brain.THINKING_TIMEOUT_S", 0.05)
    bus = EventBus()
    face = FakeFace(Config().face, bus)
    _, task = await feedback(bus, face)
    bus.publish(SpeechEnded(t_utc=1.0))
    await asyncio.sleep(0.2)
    assert eye_modes(face)[-1] == "auto"
    task.cancel()


async def test_gestures_give_tracking_back(monkeypatch):
    monkeypatch.setattr("petd.brain.expressions.GLANCE_HOLD_S", 0.05)
    face = FakeFace(Config().face, EventBus())
    expressions = Expressions(face)
    await expressions.nod()
    assert servo_modes(face) == ["manual", "track"]

    await expressions.glance("left")
    assert servo_modes(face)[-1] == "manual"
    await expressions.shake()               # during the glance's hold: still tracking after
    await asyncio.sleep(0.1)
    assert servo_modes(face)[-1] == "track"

    face.commands.clear()                   # tracking off by choice stays off
    await face.set_servo(mode="manual")
    await expressions.nod()
    assert servo_modes(face) == ["manual", "manual"]
