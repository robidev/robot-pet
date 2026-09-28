import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

from petd.brain.brain import Brain
from petd.brain.expressions import Expressions
from petd.brain.tags import Action, SpeechStreamParser
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


class SlowFace(FakeFace):
    """Every call takes `delay_s` and, if `fail`, then raises (an offline face)."""

    def __init__(self, delay_s=0.2, fail=False):
        super().__init__(Config().face, EventBus())
        self.delay_s, self.fail = delay_s, fail

    async def _slow(self):
        await asyncio.sleep(self.delay_s)
        if self.fail:
            raise ConnectionError("no route to host")

    async def set_eye_mode(self, mode):
        await self._slow()
        await super().set_eye_mode(mode)

    async def set_eye(self, x, y=0.0, aperture=1.0):
        await self._slow()
        await super().set_eye(x, y, aperture)

    async def set_servo(self, **kw):
        await self._slow()
        await super().set_servo(**kw)


async def test_a_slow_head_never_holds_up_the_words():
    face = SlowFace(delay_s=0.2)
    spoken = []
    pet = SimpleNamespace(cfg=Config(), face=face, bus=EventBus(),
                          speaker=SimpleNamespace(begin=lambda: SimpleNamespace(add=spoken.append)))
    brain = Brain(pet, backend=None)
    started = time.monotonic()
    utterance = None
    for piece in SpeechStreamParser().feed("[nod] Yes. [emote:happy] Of course. ") + [Action("rest")]:
        utterance = await brain._emit(piece, utterance, [])
    assert spoken == ["Yes.", "Of course."]
    assert time.monotonic() - started < 0.05

    # Meanwhile the head does all of it, in order: the nod, the emote, then rest.
    await asyncio.sleep(0.2 * 9 + 0.66 + 0.5)
    assert servo_modes(face) == ["manual", "track"]
    assert eye_modes(face) == ["manual", "auto"]
    assert face.commands[-1] == ("eye_mode", "auto")


async def test_a_failed_expression_is_forgotten():
    face = SlowFace(delay_s=0.0, fail=True)
    expressions = Expressions(face)
    expressions.fire(Action("emote", "happy"))
    await asyncio.sleep(0.01)
    assert face.commands == []
    face.fail = False
    expressions.fire(Action("rest"))
    await asyncio.sleep(0.05)
    assert eye_modes(face) == ["auto"]      # the emote failed; the rest after it still ran


async def test_an_unreachable_face_is_not_queued_for():
    face = SlowFace(delay_s=0.0)
    face._state = replace(face.state, reachable=False)
    expressions = Expressions(face)
    for _ in range(3):
        expressions.fire(Action("nod"))
    await asyncio.sleep(0.05)
    assert face.commands == [] and not expressions._tasks
