"""E5b: outings, with the fakes and the room's real map (PLAN.md Step 7)."""

import asyncio
import json
import math
import random
import time
from pathlib import Path

import pytest

from petd.app import App
from petd.behavior.explore import pick_spot
from petd.brain.prompt import memory_sections
from petd.io.vacuum import MapPose
from petd.spatial.mapgeo import Grid
from tests.test_initiative import NOON, StubBrain, at, config, settle

FIXTURES = Path(__file__).parent / "fixtures"
ROOM = json.loads((FIXTURES / "valetudo_map_2026-09-26.json").read_text())
IN_FRONT_OF_THE_DOCK = (2562.0, 2601.0)     # open floor (test_mapgeo)


@pytest.fixture
async def pet(tmp_path):
    cfg = config()
    cfg.memory.observations_dir = str(tmp_path / "observations")
    app = App(cfg, fake=True)
    await app.start()
    app.brain = StubBrain()
    app.initiative.clock = lambda: NOON
    app.dock.poll_s = 0.02
    app.dock.start_grace_s = 0.05
    app.vacuum.go_to_s = 0.02
    app.vacuum._map = ROOM
    app.outings.check_every_s = 0.02
    try:
        yield app
    finally:
        await app.close()


def out_in_the_room(pet, level=80):
    pet.vacuum._update(status="idle", battery_level=level, pose=MapPose(*IN_FRONT_OF_THE_DOCK, 90.0))


def on_its_own(pet):
    """A turn the pet started itself, with the switch on and someone just seen."""
    pet.initiative.set(True)
    pet.initiative.last_seen_at = NOON
    pet.brain.turn_kind = "initiative"


async def arrived(pet):
    await asyncio.wait_for(pet.motion_task, 3)
    await settle()


def go_tos(pet):
    return [c[1:] for c in pet.vacuum.commands if c[0] == "go_to"]


# --- picking a spot ---------------------------------------------------------------------

def test_a_spot_is_open_floor_a_hop_away_from_the_robot_and_the_last_spots():
    grid = Grid.from_valetudo(ROOM)
    rng = random.Random(1)
    avoid = []
    for _ in range(5):
        spot = pick_spot(grid, IN_FRONT_OF_THE_DOCK, avoid, 40, 100, rng)
        assert spot is not None
        assert grid.is_floor(*spot) and grid.clearance(*spot) >= 40
        assert math.dist(spot, IN_FRONT_OF_THE_DOCK) >= 100
        assert all(math.dist(spot, earlier) >= 100 for earlier in avoid)
        avoid.append(spot)


def test_no_spot_when_nothing_is_that_clear():
    grid = Grid.from_valetudo(ROOM)
    assert pick_spot(grid, IN_FRONT_OF_THE_DOCK, [], 1000, 100, random.Random(1)) is None


# --- the explore tool ---------------------------------------------------------------------

async def test_asked_by_a_person_it_is_just_a_trip(pet):
    out_in_the_room(pet)
    pet.brain.turn_kind = "heard"
    result = await pet.tools.call("explore", {})
    assert not result.is_error and "on my way to a spot" in result.text
    await arrived(pet)
    assert pet.outings.current is None and pet.initiative.outings_last_hour() == 0
    kind, text = pet.brain.told[-1]
    assert kind == "event" and text.startswith("arrived at a spot") and "outing" not in text
    assert len(go_tos(pet)) == 1 and pet.outings.recent_spots == [tuple(go_tos(pet)[0])]


async def test_on_its_own_it_needs_the_switch(pet):
    out_in_the_room(pet)
    pet.brain.turn_kind = "initiative"
    result = await pet.tools.call("explore", {})
    assert result.is_error and "switched off" in result.text
    assert not go_tos(pet)


async def test_an_outing_is_three_spots_then_home(pet):
    out_in_the_room(pet)
    on_its_own(pet)
    for n in (1, 2, 3):
        result = await pet.tools.call("explore", {})
        assert not result.is_error and f"spot {n} of 3" in result.text
        await arrived(pet)
        kind, text = pet.brain.told[-1]
        assert kind == "initiative" and f"spot {n} of 3" in text
    assert "last one" in pet.brain.told[-1][1]
    spots = go_tos(pet)
    assert all(math.dist(a, b) >= 100 for i, a in enumerate(spots) for b in spots[:i])
    result = await pet.tools.call("explore", {})
    assert result.is_error and "time to go home" in result.text
    assert pet.initiative.outings_last_hour() == 1


async def test_the_hours_second_outing_gets_all_its_spots(pet):
    out_in_the_room(pet)
    on_its_own(pet)
    pet.initiative.outing_started(NOON - 30 * 60)
    for n in (1, 2):
        result = await pet.tools.call("explore", {})
        assert not result.is_error, result.text
        await arrived(pet)
    assert pet.initiative.outings_last_hour() == 2
    pet.outings.current = None                          # that outing over
    result = await pet.tools.call("explore", {})
    assert result.is_error and "2 times this hour" in result.text


# --- the watchdog -------------------------------------------------------------------------

async def an_outing(pet):
    out_in_the_room(pet)
    on_its_own(pet)
    await pet.tools.call("explore", {})
    await arrived(pet)
    pet.brain.turn_kind = None
    assert pet.outings.current is not None


@pytest.mark.parametrize("ending,why", [
    ("switch", "my initiative was switched off"),
    ("quiet hours", "quiet hours have begun (20:00)"),
    ("nobody", "nobody has been around for 30 minutes"),
])
async def test_an_outing_ends_and_heads_home(pet, ending, why):
    await an_outing(pet)
    if ending == "switch":
        pet.initiative.set(False)
    elif ending == "quiet hours":
        pet.initiative.clock = lambda: at(20, 0)
    else:
        pet.initiative.last_seen_at = NOON - 31 * 60
    await settle(0.1)
    assert pet.outings.current is None
    assert pet.brain.told[-1] == ("event", f"My outing is over: {why}. I'm already driving back to my dock.")
    await asyncio.wait_for(pet.motion_task, 2)
    assert pet.vacuum.state.docked


async def test_a_low_battery_on_an_outing_is_told_once(pet):
    await an_outing(pet)
    told = len(pet.brain.told)
    pet.vacuum._update(battery_level=30)
    await settle(0.1)
    assert pet.brain.told[told:] == [
        ("event", "My outing is over: my battery is down to 30%. I'm already driving back to my dock.")]


async def test_a_person_taking_over_ends_the_outing_where_it_stands(pet):
    await an_outing(pet)
    pet.brain.turn_kind = "heard"
    result = await pet.tools.call("move", {"cm": 10})
    assert not result.is_error
    assert pet.outings.current is None
    pet.initiative.set(False)
    await settle(0.1)
    assert not any(c[0] == "dock" for c in pet.vacuum.commands)


async def test_back_on_the_dock_the_outing_is_over(pet):
    await an_outing(pet)
    pet.vacuum._update(status="docked")
    await settle()
    assert pet.outings.current is None
    assert pet.brain.told[-1][0] == "initiative"        # the arrival, nothing since


# --- what it saw -------------------------------------------------------------------------

async def test_an_observation_keeps_the_last_look_and_comes_back_in_the_prompt(pet, tmp_path):
    out_in_the_room(pet)
    pet.last_photo = (time.time(), b"\xff\xd8 a photo")
    result = await pet.tools.call("note_observation", {"caption": "a new plant on the windowsill"})
    assert not result.is_error
    (seen,) = pet.db.observations()
    assert seen.caption == "a new plant on the windowsill" and seen.x is not None
    assert Path(seen.jpeg_path).read_bytes() == b"\xff\xd8 a photo"
    assert "# Things I noticed lately" in memory_sections(pet)
    assert "a new plant on the windowsill" in memory_sections(pet)


async def test_an_old_photo_is_not_kept_and_only_the_newest_photos_stay(pet, tmp_path):
    pet.cfg.memory.observations_keep_photos = 2
    pet.last_photo = (time.time() - 600, b"old")
    await pet.tools.call("note_observation", {"caption": "nothing much"})
    assert pet.db.observations()[0].jpeg_path is None
    for n in range(3):
        pet.last_photo = (time.time(), b"photo %d" % n)
        await pet.tools.call("note_observation", {"caption": f"thing {n}"})
        await asyncio.sleep(0.02)                       # distinct modification times
    kept = sorted(p.name for p in (tmp_path / "observations").glob("*.jpg"))
    assert len(kept) == 2
