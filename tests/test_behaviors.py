"""E4: coming over to someone and looking around for them, with the fakes (PLAN.md Step 5)."""

import asyncio
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from petd.app import App
from petd.brain.expressions import Expressions
from petd.config import Config
from petd.io.face import Face
from petd.io.vacuum import MapPose


def face(cx=0.5, h=0.3, cy=0.45):
    return Face(-1, 0.9, cx - h * 0.4, cy - h / 2, cx + h * 0.4, cy + h / 2)


def config() -> Config:
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.face.search_speed_deg_s = 1000.0        # a pass in ~0.2 s
    cfg.face.search_naming_wait_s = 0.3
    cfg.person.look_settle_s = 0.0
    m = cfg.motion
    m.warmup_s, m.resend_s, m.settle_s, m.idle_disarm_s = 0.0, 0.02, 0.1, 0.3
    m.max_turn_rate_deg_s = 240.0
    return cfg


@pytest.fixture
async def pet(monkeypatch):
    monkeypatch.setattr("petd.behavior.search.SEARCH_STEP_S", 0.01)
    monkeypatch.setattr("petd.behavior.approach.LOOK_GAP_S", 0.0)
    cfg = config()
    app = App(cfg, fake=True)
    await app.start()
    told: list = []
    app.brain = SimpleNamespace(expressions=Expressions(app.face, cfg.calibration),
                                tell=lambda text, kind="heard", speaker=None: told.append((kind, text)),
                                note=lambda text: told.append(("note", text)), close=nothing)
    app.told = told
    app.vacuum.go_to_s = 0.02
    try:
        yield app
    finally:
        await app.close()


async def nothing():
    pass


async def show(app, *faces):
    app.face.show(*faces)
    await asyncio.sleep(0.01)                   # the adapter takes it in


def off_the_dock(app, x=2560.0, y=2700.0, angle=90.0):
    app.vacuum._update(status="idle", pose=MapPose(x, y, angle))


async def finished(app, timeout_s=10.0):
    for _ in range(int(timeout_s / 0.01)):
        if not app.moving:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the behaviour didn't end")


def go_tos(app):
    return [c[1:] for c in app.vacuum.commands if c[0] == "go_to"]


# --- the heading -----------------------------------------------------------------

async def test_a_turn_on_the_spot_turns_the_heading_until_valetudo_drives(pet):
    off_the_dock(pet, angle=90.0)
    await asyncio.sleep(0.01)
    result = await pet.motion.turn_by(60)            # to the left: Valetudo's angle shrinks
    assert result.ok
    assert pet.robot_pose().angle == pytest.approx(90.0 - result.done, abs=1)
    assert pet.vacuum.state.pose.angle == 90.0         # Valetudo's own stays where it was
    await pet.motion.move_by(-5)                       # backing up keeps it
    assert pet.robot_pose().angle == pytest.approx(90.0 - result.done, abs=1)
    await pet.vacuum.go_to(2600, 2700)                 # Valetudo drives: its heading again
    await asyncio.sleep(0.05)
    assert pet.heading.override is None and pet.robot_pose().angle == 90.0


# --- search_for_person --------------------------------------------------------------

async def test_a_search_on_the_dock_stops_at_a_face_and_follows_it(pet):
    await pet.face.set_servo(mode="manual", pan_deg=60.0, tilt_deg=90.0)
    result = await pet.tools.call("search_for_person", {})
    assert "Nothing found yet" in result.text and pet.moving
    await asyncio.sleep(0.01)
    assert pet.brain.expressions.owner == "search"
    for _ in range(200):                                # to the nearer end (0) first, then across
        if pet.face.state.pan_deg > 100.0:
            break
        await asyncio.sleep(0.005)
    assert pet.face.state.tilt_deg == pet.cfg.face.search_tilt_deg
    pet.face._state = replace(pet.face.state, pan_deg=130.0)
    pet.face.show(face())                               # seen in a frame taken at pan 130
    pet.face._state = replace(pet.face.state, pan_deg=150.0)   # the head has moved on
    await finished(pet)
    assert (pet.face.state.pan_deg, pet.face.state.servo_mode) == (130.0, "track")
    kind, text = pet.told[-1]
    assert kind == "event" and "found a face" in text and "don't recognise" in text
    assert pet.brain.expressions.owner is None

    result = await pet.tools.call("search_for_person", {})    # a face in view: no search
    assert "already see a face" in result.text and not pet.moving


async def test_a_search_on_the_dock_finds_nobody_and_says_it_could_not_turn(pet):
    pet.cfg.face.search_sweeps = 1
    await pet.tools.call("search_for_person", {})
    await finished(pet)
    assert (pet.face.state.pan_deg, pet.face.state.servo_mode) == (pet.cfg.calibration.pan_forward_deg, "track")
    kind, text = pet.told[-1]
    assert kind == "event" and "didn't find anyone" in text and "on my dock" in text
    assert not [c for c in pet.vacuum.commands if c[0] == "manual_move"]


async def test_a_search_off_the_dock_turns_the_body_between_passes(pet):
    off_the_dock(pet)
    await pet.tools.call("search_for_person", {})
    await finished(pet)
    assert "turned 2 times" in pet.told[-1][1]
    assert pet.robot_pose().angle == pytest.approx((90.0 - 240.0) % 360, abs=10)


async def test_a_search_by_name_passes_someone_else_by(pet):
    robin = pet.db.add_person("Robin")
    claudia_at, robin_at = 40.0, 140.0
    result = await pet.tools.call("search_for_person", {"name": "robin"})
    assert "search for Robin started" in result.text

    async def stage():
        shown = set()
        while pet.moving:
            pan = pet.face.state.pan_deg
            if pan <= claudia_at + 5 and "claudia" not in shown:       # passing her (to 0 first)
                shown.add("claudia")
                pet.face._state = replace(pet.face.state, pan_deg=claudia_at)
                pet.face.show(face())                   # nobody recognises her: not Robin
                await asyncio.sleep(0.05)
                pet.face.show()
            elif pan >= robin_at and "robin" not in shown and "claudia" in shown:
                shown.add("robin")
                pet.face._state = replace(pet.face.state, pan_deg=robin_at)
                pet.face.show(face())
                pet.people.recognized(robin.id, 0.0, 0.8, centre=(0.5, 0.45))
            await asyncio.sleep(0.005)
    await asyncio.wait_for(stage(), 10)
    kind, text = pet.told[-1]
    assert kind == "event" and "found Robin" in text, text
    assert pet.face.state.pan_deg == robin_at


async def test_the_head_is_the_behaviours_while_it_runs_and_stop_ends_it(pet):
    pet.cfg.face.search_speed_deg_s = 20.0              # slow enough to catch it running
    await pet.tools.call("search_for_person", {})
    await asyncio.sleep(0.05)
    refused = await pet.tools.call("look_direction", {"direction": "left"})
    assert refused.is_error and "busy" in refused.text
    assert (await pet.tools.call("track_faces", {"on": False})).is_error
    await pet.brain.expressions.glance("left")          # only the eye
    assert not pet.brain.expressions._glance_pending()
    await pet.stop_everything("test")
    assert not pet.moving and pet.brain.expressions.owner is None
    assert pet.face.state.servo_mode == "track"


async def test_search_by_name_needs_someone_known():
    cfg = config()
    app = App(cfg, fake=True)
    await app.start()
    try:
        result = await app.tools.call("search_for_person", {"name": "Zorro"})
        assert result.is_error and "don't know anyone called Zorro" in result.text
    finally:
        await app.close()


# --- approach_person ---------------------------------------------------------------

async def test_coming_over_from_far_drives_part_way_looks_again_and_stops_in_front(pet):
    off_the_dock(pet, angle=180.0)
    await show(pet, face(h=0.07))                       # small: ~3.5 m off
    result = await pet.tools.call("approach_person", {})
    assert "coming over to them" in result.text
    await finished(pet)
    first, last = go_tos(pet)
    kind, text = pet.told[-1]
    assert kind == "event" and "I came over" in text, text
    assert pet.brain.expressions.owner is None and pet.face.state.servo_mode == "track"
    # The second look is from where the first leg ended, with the face as small
    # as before (the fake doesn't change it): the last leg stops standoff short.
    start = (2560.0, 2700.0)
    assert math.dist(start, first) > 50 and math.dist(first, last) > 50


async def test_coming_over_from_close_stays_put(pet):
    off_the_dock(pet)
    await show(pet, face(h=0.22))                       # ~1.1 m
    await pet.tools.call("approach_person", {})
    await finished(pet)
    assert not go_tos(pet)
    assert "stayed put" in pet.told[-1][1]


async def test_coming_over_needs_someone_in_view(pet):
    robin = pet.db.add_person("Robin")
    result = await pet.tools.call("approach_person", {})
    assert result.is_error and "search_for_person first" in result.text
    await show(pet, face())
    result = await pet.tools.call("approach_person", {"name": "Robin"})
    assert result.is_error and "can't see Robin" in result.text
    pet.people.recognized(robin.id, 0.0, 0.8, centre=(0.5, 0.45))
    pet.vacuum._update(battery_level=20)                # on the dock, too low to leave
    result = await pet.tools.call("approach_person", {"name": "Robin"})
    assert result.is_error and "too low to leave" in result.text


async def test_stop_while_coming_over(pet):
    off_the_dock(pet)
    pet.vacuum.go_to_s = 5.0                            # a long drive
    await show(pet, face(h=0.07))
    await pet.tools.call("approach_person", {})
    for _ in range(200):
        if go_tos(pet):
            break
        await asyncio.sleep(0.01)
    await pet.stop_everything("voice")
    assert not pet.moving and pet.brain.expressions.owner is None
    assert ("stop",) in pet.vacuum.commands


# --- what the brain is told ----------------------------------------------------------

async def test_senses_name_faces_by_the_pcs_recognition_not_the_heads_slots(pet):
    from petd.memory.recognition import Track
    robin = pet.db.add_person("Robin")
    await show(pet, Face(0, 0.9, 0.4, 0.3, 0.6, 0.6), face(cx=0.85))   # the head says "slot 0"
    pet.recognizer.tracks = [Track(centre=(0.5, 0.45), first_seen=0.0, person_id=robin.id)]
    seen = json.loads((await pet.tools.call("get_senses", {})).text)
    assert [s["who"] for s in seen["sees"]] == ["Robin", "not recognised yet"]
    assert "known_faces_stored" not in seen["head"]


# --- the SLAM pose -------------------------------------------------------------------

def test_a_slam_estimate_is_valetudos_frame_and_angle():
    from petd.io.slam import parse
    docked = parse("8682.625 estimate 0.008 0.090 -1.581")      # on the dock, 2026-10-11
    assert (round(docked.x), round(docked.y)) == (2561, 2551)    # Valetudo said (2561, 2550)
    assert docked.angle == pytest.approx(180.6, abs=0.1)         # away from the dock
    assert parse("8683.088 lock") is None and parse("1 estimate x y z") is None


async def test_the_follower_keeps_the_newest_estimate():
    from petd.config import VacuumConfig
    from petd.io.slam import SlamFollower
    follower = SlamFollower(VacuumConfig())
    script = "printf '1.0 estimate 0 0 0\\n1.1 pause\\n1.2 estimate 1.0 -0.5 1.5708\\n'"
    await follower._follow(["sh", "-c", script])
    pose = follower.pose()
    assert (round(pose.x), round(pose.y)) == (2660, 2610)
    assert min(pose.angle, 360 - pose.angle) < 0.1         # facing up the map


async def test_the_heading_is_the_slams_when_there_is_one(pet):
    off_the_dock(pet, angle=220.0)
    pet.heading.slam = SimpleNamespace(pose=lambda: MapPose(2600.0, 2650.0, 131.0))
    await pet.motion.turn_by(30)                        # the measured turn is only the fallback
    assert pet.robot_pose() == MapPose(2600.0, 2650.0, 131.0)
    pet.heading.slam = SimpleNamespace(pose=lambda: None)   # gone: the turn, from the SLAM's 131
    assert pet.robot_pose().angle == pytest.approx(131.0 - 30.0, abs=6)


async def test_coming_over_looks_again_when_the_first_leg_stops_short(pet):
    off_the_dock(pet, angle=180.0)
    await show(pet, face(h=0.07))
    real_go_to, legs = pet.vacuum.go_to, []

    async def blocked_first(x, y):
        legs.append((x, y))
        if len(legs) == 1:                              # the couch: 60 cm short
            here = pet.vacuum.state.pose
            dx, dy = x - here.x, y - here.y
            k = max(0.0, 1 - 60 / math.hypot(dx, dy))
            return await real_go_to(here.x + dx * k, here.y + dy * k)
        return await real_go_to(x, y)
    pet.vacuum.go_to = blocked_first
    await pet.tools.call("approach_person", {})
    await finished(pet)
    assert len(legs) == 2
    assert "I came over" in pet.told[-1][1], pet.told
