"""E5c: nudges, quiet hours, perking up and idle glances, with the fakes (PLAN.md Step 7)."""

import asyncio

import pytest

from petd.app import App
from petd.behavior.nudges import Nudges
from petd.brain.expressions import Expressions
from petd.events import MotionDetected
from petd.io.vacuum import MapPose
from tests.test_initiative import NOON, StubBrain, at, config, settle, someone_in_view


@pytest.fixture
async def pet():
    cfg = config()
    app = App(cfg, fake=True)
    await app.start()
    app.brain = StubBrain()
    fired = []
    app.brain.expressions = Expressions(app.face, cfg.calibration)
    app.brain.expressions.fire = fired.append
    app.fired = fired
    clock = {"now": NOON}
    app.initiative.clock = lambda: clock["now"]
    app.clock = clock
    app.nudges = Nudges(app)
    app.nudges.tick_s = 3600                    # the tests tick by hand
    await app.nudges.start()
    try:
        yield app
    finally:
        await app.close()


def switched_on(pet):
    pet.initiative.set(True)
    pet.brain.notes.clear()


async def someone_here_for(pet, minutes):
    someone_in_view(pet)
    await settle()
    pet.nudges.in_view_since = NOON - minutes * 60


def idle_for(pet, minutes, seen_min_ago=5):
    pet.nudges.last_active_at = NOON - minutes * 60
    pet.initiative.last_seen_at = NOON - seen_min_ago * 60


def nudged(pet):
    return [text for kind, text in pet.brain.told if kind == "initiative"]


# --- the switch ------------------------------------------------------------------------

async def test_with_the_switch_off_nothing_is_started(pet):
    await someone_here_for(pet, 10)
    idle_for(pet, 60)
    assert await pet.nudges.tick() is None
    pet.clock["now"] = at(20, 0)
    assert await pet.nudges.tick() is None
    pet.bus.publish(MotionDetected(active=True, pan_deg=80.0, tilt_deg=100.0, device_utc=None))
    await settle()
    assert pet.brain.told == [] and pet.fired == []


# --- social ------------------------------------------------------------------------------

async def test_someone_around_for_a_while_gets_a_word(pet):
    switched_on(pet)
    await someone_here_for(pet, 3)
    assert await pet.nudges.tick() == "social"
    (text,) = nudged(pet)
    assert "has been in view for about 3 minutes" in text and "saying nothing is fine" in text
    assert await pet.nudges.tick() is None                  # at most every 15 minutes
    pet.clock["now"] = NOON + 16 * 60
    assert await pet.nudges.tick() == "social"


async def test_no_word_too_soon_mid_conversation_or_after_be_quiet(pet):
    switched_on(pet)
    await someone_here_for(pet, 1)
    assert await pet.nudges.tick() is None                  # only just arrived
    pet.nudges.in_view_since = NOON - 5 * 60
    pet.nudges.last_heard_at = NOON - 10 * 60
    assert await pet.nudges.tick() is None                  # they talked to it 10 min ago
    pet.nudges.last_heard_at = None
    pet.brain.busy = True
    assert await pet.nudges.tick() is None
    pet.brain.busy = False
    pet.initiative.mute(NOON)
    assert await pet.nudges.tick() is None
    assert nudged(pet) == []


# --- restless ----------------------------------------------------------------------------

async def test_nothing_to_do_for_40_minutes_invites_an_outing(pet):
    switched_on(pet)
    idle_for(pet, 39)
    assert await pet.nudges.tick() is None
    idle_for(pet, 41)
    assert await pet.nudges.tick() == "restless"
    (text,) = nudged(pet)
    assert "Nothing has needed me for 41 minutes" in text and "explore" in text
    assert "Nobody is in view (someone was, 5 minutes ago)" in text
    assert await pet.nudges.tick() is None                  # the nudge itself counts as something


async def test_no_outing_invited_when_the_rules_say_no(pet):
    switched_on(pet)
    idle_for(pet, 60, seen_min_ago=31)
    assert await pet.nudges.tick() is None                  # nobody around lately
    idle_for(pet, 60)
    pet.initiative.outing_started(NOON - 20 * 60)
    pet.initiative.outing_started(NOON - 10 * 60)
    assert await pet.nudges.tick() is None                  # two this hour
    pet.initiative.outings.clear()
    pet.clock["now"] = at(21, 0)
    pet.nudges._quiet = True
    idle_for(pet, 60)
    pet.nudges.last_active_at = at(21, 0) - 60 * 60
    assert await pet.nudges.tick() is None                  # quiet hours
    assert nudged(pet) == []


# --- quiet hours -------------------------------------------------------------------------

async def test_goodnight_and_good_morning(pet):
    switched_on(pet)
    pet.clock["now"] = at(19, 59)
    await pet.nudges.tick()
    pet.clock["now"] = at(20, 0)
    assert await pet.nudges.tick() == "goodnight"
    assert "Quiet hours have begun (20:00 until 09:00)" in nudged(pet)[-1]
    assert "go_home" not in nudged(pet)[-1]                 # docked
    assert await pet.nudges.tick() is None                  # once
    pet.clock["now"] = at(9, 0) + 24 * 3600
    assert await pet.nudges.tick() == "morning"


async def test_goodnight_off_the_dock_says_go_home(pet):
    switched_on(pet)
    pet.vacuum._update(status="idle", pose=MapPose(2700, 2650, 90.0))
    pet.clock["now"] = at(19, 59)
    await pet.nudges.tick()
    pet.clock["now"] = at(20, 0)
    await pet.nudges.tick()
    assert "go home now (go_home)" in nudged(pet)[-1]


async def test_no_goodnight_when_the_outing_watchdog_just_said_so(pet):
    switched_on(pet)
    pet.clock["now"] = at(19, 59)
    await pet.nudges.tick()
    pet.clock["now"] = at(20, 0)
    pet.outings.last_end = (at(20, 0) - 30, "quiet hours have begun (20:00)")
    assert await pet.nudges.tick() is None


# --- perking up and idle glances -------------------------------------------------------------

async def test_motion_with_nobody_in_view_perks_it_up_now_and_then(pet):
    switched_on(pet)
    motion = MotionDetected(active=True, pan_deg=80.0, tilt_deg=100.0, device_utc=None)
    pet.bus.publish(motion)
    await settle()
    assert [(a.kind, a.value) for a in pet.fired] == [("emote", "curious"), ("look", "up")]
    pet.bus.publish(motion)
    await settle()
    assert len(pet.fired) == 2                              # not again within a minute
    pet.clock["now"] = NOON + 61
    someone_in_view(pet)
    await settle()
    pet.bus.publish(motion)
    await settle()
    assert len(pet.fired) == 2                              # a face in view: tracking has it


async def test_with_nobody_about_the_head_glances_every_few_minutes(pet):
    switched_on(pet)
    pet.nudges.last_glance_at = NOON - 2 * 60
    await pet.nudges.tick()
    assert pet.fired == []
    pet.nudges.last_glance_at = NOON - 4 * 60
    await pet.nudges.tick()
    assert [a.kind for a in pet.fired] == ["look"] and pet.fired[0].value in ("left", "right", "up")
    pet.clock["now"] = at(22, 0)
    pet.nudges._quiet = True
    pet.nudges.last_glance_at = at(22, 0) - 10 * 60
    await pet.nudges.tick()
    assert len(pet.fired) == 1                              # asleep in quiet hours
