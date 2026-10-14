"""E5a: the initiative switch, the house rules and low battery, with the fakes (PLAN.md Step 7)."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from petd.api.server import create_app
from petd.app import App
from petd.behavior.initiative import in_quiet_hours
from petd.config import Config, ConfigError
from petd.events import InitiativeChanged, LowBattery
from petd.io.face import Face
from petd.io.vacuum import MapPose


def at(hour, minute=0):
    """Epoch seconds for today at hour:minute, local time."""
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, minute, 0, 0, 0, -1))


NOON = at(12)


def config() -> Config:
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.recognition.enabled = False
    cfg.face.faces_lost_debounce_s = 0.05
    m = cfg.motion
    m.warmup_s, m.resend_s, m.settle_s, m.idle_disarm_s = 0.0, 0.02, 0.05, 0.3
    m.dock_timeout_s = 0.3
    m.low_battery_resume_s = 0.2
    return cfg


class StubBrain:
    def __init__(self):
        self.told, self.notes = [], []
        self.turn_kind = None
        self.expressions = None
        self.busy = False

    def tell(self, text, kind="heard", speaker=None):
        self.told.append((kind, text))

    def note(self, text):
        self.notes.append(text)

    def hush(self):
        pass

    async def close(self):
        pass


@pytest.fixture
async def pet():
    app = App(config(), fake=True)
    await app.start()
    app.brain = StubBrain()
    app.initiative.clock = lambda: NOON
    app.dock.poll_s = 0.02
    app.dock.start_grace_s = 0.05
    app.vacuum.go_to_s = 0.02
    try:
        yield app
    finally:
        await app.close()


async def settle(s=0.05):
    await asyncio.sleep(s)


def off_the_dock(pet, level=80, x=2800.0, y=2700.0):
    pet.vacuum._update(status="idle", battery_level=level, pose=MapPose(x, y, 90.0))


def someone_in_view(pet):
    pet.face.show(Face(-1, 0.9, 0.4, 0.3, 0.6, 0.6))


# --- quiet hours ------------------------------------------------------------------

@pytest.mark.parametrize("hour,minute,quiet", [
    (19, 59, False), (20, 0, True), (23, 30, True), (0, 0, True), (8, 59, True), (9, 0, False),
    (12, 0, False)])
def test_quiet_hours_run_past_midnight(hour, minute, quiet):
    assert in_quiet_hours(at(hour, minute), ["20:00", "09:00"]) is quiet


def test_quiet_hours_within_a_day():
    assert in_quiet_hours(at(13, 30), ["13:00", "14:00"])
    assert not in_quiet_hours(at(14, 0), ["13:00", "14:00"])


async def test_quiet_hours_that_are_not_times_stop_petd_starting():
    cfg = config()
    cfg.initiative.quiet_hours = ["8pm", "09:00"]
    app = App(cfg, fake=True)
    with pytest.raises(ConfigError):
        await app.start()
    await app.close()


# --- the switch ---------------------------------------------------------------------

async def test_the_switch_starts_off_and_the_api_turns_it_on(pet):
    changes = pet.bus.subscribe(InitiativeChanged)
    assert not pet.initiative.enabled
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(pet)),
                                 base_url="http://petd") as c:
        assert (await c.get("/status")).json()["initiative"]["on"] is False
        result = (await c.post("/initiative", json={"on": True})).json()
        assert result["on"] is True
        assert (await c.get("/status")).json()["initiative"]["on"] is True
        assert "initiative" in (await c.get("/")).text
    event = await asyncio.wait_for(changes.get(), 1)
    assert event.on and event.source == "api"
    assert "switched on" in pet.brain.notes[-1]


async def test_the_config_can_start_it_on():
    cfg = config()
    cfg.initiative.enabled = True
    app = App(cfg, fake=True)
    await app.start()
    try:
        assert app.initiative.enabled and app.initiative.greets
    finally:
        await app.close()


# --- the rules ------------------------------------------------------------------------

async def test_roaming_needs_the_switch_daytime_and_someone_seen_lately(pet):
    rules = pet.initiative
    rules.last_seen_at = NOON - 10 * 60
    assert "switched off" in rules.why_not_roam()
    rules.set(True)
    assert rules.why_not_roam() is None
    assert "quiet hours" in rules.why_not_roam(at(20, 0))
    assert rules.why_not_roam(NOON + 50 * 60) is not None       # an hour since anyone was seen
    rules.last_seen_at = NOON - 29 * 60
    assert rules.why_not_roam() is None
    rules.last_seen_at = NOON - 31 * 60
    assert "nobody has been around for 30 minutes" in rules.why_not_roam()
    someone_in_view(pet)
    await settle()
    assert rules.why_not_roam() is None                          # in view now


async def test_roaming_is_at_most_twice_an_hour(pet):
    rules = pet.initiative
    rules.set(True)
    rules.last_seen_at = NOON
    rules.outing_started(NOON - 50 * 60)
    rules.outing_started(NOON - 10 * 60)
    assert "2 times this hour" in rules.why_not_roam()
    assert rules.why_not_roam(NOON + 11 * 60) is None             # the first is an hour old


async def test_roaming_minds_the_battery(pet):
    rules = pet.initiative
    rules.set(True)
    rules.last_seen_at = NOON
    pet.vacuum._update(battery_level=39)                          # docked
    assert "too low to leave the dock" in rules.why_not_roam()
    pet.vacuum._update(battery_level=40)
    assert rules.why_not_roam() is None
    off_the_dock(pet, level=30)
    await settle()
    assert "heading home" in rules.why_not_roam()


async def test_speaking_up_needs_someone_in_view_and_not_too_often(pet):
    rules = pet.initiative
    rules.set(True)
    assert rules.why_not_remark() == "nobody in view"
    someone_in_view(pet)
    await settle()
    assert rules.why_not_remark() is None
    assert rules.why_not_remark(at(21, 0)) == "quiet hours"
    rules.remarked(NOON - 14 * 60)
    assert rules.why_not_remark() == "spoke up recently"
    rules.remarked(NOON - 16 * 60)
    assert rules.why_not_remark() is None
    pet.brain.busy = True
    assert rules.why_not_remark() == "mid-conversation"
    pet.brain.busy = False
    rules.mute(NOON)
    assert rules.why_not_remark().startswith("told to be quiet")
    assert rules.why_not_remark(NOON + 61 * 60) is None


async def test_be_quiet_mutes_speaking_up(pet):
    from petd.behavior.converse import Listener
    pet.listener = Listener(pet)
    pet.initiative.clock = time.time
    await pet.listener.reflex("quiet", "be quiet")
    assert pet.initiative.muted_until > time.time() + 59 * 60


# --- motion in a turn the pet started itself ----------------------------------------------

async def test_a_self_started_turn_moves_only_within_the_rules(pet):
    off_the_dock(pet)
    pet.brain.turn_kind = "initiative"
    result = await pet.tools.call("move", {"cm": 10})
    assert result.is_error and "not now: my initiative is switched off" in result.text
    pet.initiative.set(True)
    pet.initiative.last_seen_at = NOON
    result = await pet.tools.call("move", {"cm": 10})
    assert not result.is_error

    pet.initiative.clock = lambda: at(21, 0)
    result = await pet.tools.call("search_for_person", {})
    assert result.is_error and "quiet hours" in result.text


async def test_what_a_person_asks_for_is_never_held_back(pet):
    off_the_dock(pet)
    pet.initiative.clock = lambda: at(21, 0)                     # quiet hours, switch off
    pet.brain.turn_kind = "heard"
    result = await pet.tools.call("move", {"cm": 10})
    assert not result.is_error


# --- low battery -------------------------------------------------------------------------

async def test_at_30_percent_the_brain_is_told_once(pet):
    off_the_dock(pet, level=31)
    await settle()
    assert pet.brain.told == []
    pet.vacuum._update(battery_level=30)
    await settle()
    kind, text = pet.brain.told[-1]
    assert kind == "event" and "30%" in text and "go_home" in text
    pet.vacuum._update(battery_level=29)
    await settle()
    assert len(pet.brain.told) == 1
    assert not any(c[0] in ("go_to", "dock") for c in pet.vacuum.commands)


async def test_at_22_percent_it_is_sent_home(pet):
    off_the_dock(pet, level=23)
    await settle()
    events = pet.bus.subscribe(LowBattery)
    pet.vacuum._update(battery_level=22)
    event = await asyncio.wait_for(events.get(), 1)
    assert (event.level, event.action) == (22, "docking")
    await asyncio.wait_for(pet.motion_task, 2)
    assert pet.vacuum.state.docked
    assert "already driving back" in pet.brain.told[-1][1]


async def test_docking_starts_a_new_discharge(pet):
    off_the_dock(pet, level=30)
    await settle()
    pet.vacuum._update(status="docked", battery_level=90)
    await settle()
    off_the_dock(pet, level=30)
    await settle()
    assert len([t for k, t in pet.brain.told if "30%" in t]) == 2


async def test_the_robots_own_return_is_left_alone(pet):
    off_the_dock(pet, level=19)
    await settle()                                  # petd sends it home itself first...
    await pet.cancel_motion()                       # ...but say that never got going
    pet.vacuum.commands.clear()

    async def a_long_motion():
        await asyncio.sleep(10)
        return "done", True
    pet.start_motion(a_long_motion(), "a long motion")
    pet.vacuum._update(status="returning")          # the robot heads home by itself
    await settle()
    assert pet.battery.own_return and not pet.moving
    assert not any(c[0] in ("stop", "pause", "go_to", "dock") for c in pet.vacuum.commands)
    assert "by itself" in pet.brain.told[-1][1]

    result = await pet.tools.call("move", {"cm": 10})
    assert result.is_error and "by itself" in result.text
    result = await pet.tools.call("go_home", {})
    assert not result.is_error and "already taking me back" in result.text

    pet.vacuum._update(status="docked")
    await settle()
    assert not pet.battery.own_return


async def test_a_stop_on_the_way_home_is_undone_a_minute_later(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2560, 2550), MapPose(2560, 2530)
    pet.vacuum.go_to_s = 5.0                        # still on its way when stopped
    off_the_dock(pet, level=28)
    await settle()
    await pet.go_home()
    await settle()
    assert pet.dock.going_home
    await pet.stop_everything("voice")
    assert "head home again" in pet.brain.notes[-1]
    pet.vacuum.go_to_s = 0.02
    await asyncio.sleep(0.4)
    assert [c[0] for c in pet.vacuum.commands].count("go_to") == 2
    assert "driving back to my dock again" in pet.brain.told[-1][1]


async def test_docked_before_the_minute_is_up_nothing_is_resent(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2560, 2550), MapPose(2560, 2530)
    pet.vacuum.go_to_s = 5.0
    off_the_dock(pet, level=28)
    await settle()
    await pet.go_home()
    await settle()
    await pet.stop_everything("voice")
    pet.vacuum._update(status="docked")             # someone put it back
    await asyncio.sleep(0.4)
    assert [c[0] for c in pet.vacuum.commands].count("go_to") == 1


async def test_a_stop_with_a_good_battery_is_just_a_stop(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2560, 2550), MapPose(2560, 2530)
    pet.vacuum.go_to_s = 5.0
    off_the_dock(pet, level=60)
    await settle()
    await pet.go_home()
    await settle()
    await pet.stop_everything("voice")
    await asyncio.sleep(0.4)
    assert [c[0] for c in pet.vacuum.commands].count("go_to") == 1


async def test_the_tail_of_its_own_trip_home_is_not_the_robots_return(pet):
    off_the_dock(pet, level=60)
    pet.dock.going_home = True
    pet.vacuum._update(status="returning")      # petd's own dock sequence...
    await settle()
    pet.dock.going_home = False                 # ...stopped, and a pose update comes in late
    pet.vacuum._update(pose=MapPose(2700, 2650, 90.0))
    await settle()
    assert not pet.battery.own_return and pet.brain.told == []
