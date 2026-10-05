import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from petd.brain.brain import Brain
from petd.brain.expressions import Expressions
from petd.brain.tags import Action, SpeechStreamParser
from petd.bus import EventBus
from petd.config import CalibrationConfig, Config
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


def poses(face):
    return [(c[2], c[3]) for c in face.commands if c[0] == "servo" and c[1] is None]


async def test_a_glance_turns_the_head_back(monkeypatch):
    # 2026-10-05: tracking came back on after [look:left], but nobody was in
    # frame any more, so the head looked at nothing for 2.5 minutes.
    monkeypatch.setattr("petd.brain.expressions.GLANCE_HOLD_S", 0.05)
    face = FakeFace(Config().face, EventBus())
    await face.set_servo(mode="track", pan_deg=80.0, tilt_deg=100.0)
    face.commands.clear()
    expressions = Expressions(face)
    await expressions.glance("left")
    await expressions.glance("right")       # a second glance within the hold: back to the first pose
    await asyncio.sleep(0.1)
    assert poses(face)[-1] == (80.0, 100.0)
    assert servo_modes(face)[-1] == "track"

    face.commands.clear()                   # not tracking: back to the pose, tracking stays off
    await face.set_servo(mode="manual")
    await expressions.glance("up")
    await expressions.nod()                 # a nod mid-glance doesn't lose the way back
    await asyncio.sleep(0.1)
    assert poses(face)[-1] == (80.0, 100.0)
    assert servo_modes(face)[-1] == "manual"


@pytest.mark.parametrize("pan_sign,tilt_sign,pan,tilt", [
    (1.0, 1.0, 110.0, 65.0),                # before 2026-10-05: pan grew to the left, lower tilt looked up
    (-1.0, -1.0, 50.0, 95.0),               # remounted: both reversed
])
async def test_left_and_up_follow_the_calibration(monkeypatch, pan_sign, tilt_sign, pan, tilt):
    monkeypatch.setattr("petd.brain.expressions.GLANCE_HOLD_S", 10.0)
    cfg = replace(Config().face, tilt_min_deg=0.0, tilt_max_deg=180.0)
    face = FakeFace(cfg, EventBus())
    await face.set_servo(mode="manual", pan_deg=80.0, tilt_deg=80.0)
    cal = replace(CalibrationConfig(), pan_sign=pan_sign, tilt_deg_per_elevation_deg=tilt_sign)
    expressions = Expressions(face, cal)
    await expressions.glance("left")
    await face.set_servo(pan_deg=80.0, tilt_deg=80.0)
    await expressions.glance("up")
    (left_pan, _), _, (_, up_tilt) = poses(face)[-3:]
    assert (left_pan, up_tilt) == (pan, tilt)
    expressions._resume.cancel()


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


async def test_a_pause_opening_the_reply_is_skipped():
    # 2026-09-28: without thinking, replies often began "[pause]", holding
    # the first sentence back 0.4 s for nothing.
    added = []
    pet = SimpleNamespace(cfg=Config(), face=FakeFace(Config().face, EventBus()), bus=EventBus(),
                          speaker=SimpleNamespace(begin=lambda: SimpleNamespace(
                              add=lambda s: added.append((s, time.monotonic())))))
    brain = Brain(pet, backend=None)
    started, utterance, said = time.monotonic(), None, []
    for piece in SpeechStreamParser().feed("[pause] Four. [pause] Five. "):
        utterance = await brain._emit(piece, utterance, said)
    (first, t1), (second, t2) = added
    assert (first, second) == ("Four.", "Five.")
    assert t1 - started < 0.05 and t2 - t1 >= 0.4     # the second pause still a beat
