import asyncio

import pytest

from petd.app import App
from petd.config import Config
from petd.io.vacuum import MapPose, VacuumState
from petd.spatial.dock import DOCK_KEY, approach_point, await_arrival


def test_the_approach_point_is_straight_out_in_front_of_the_dock():
    # The live map on 2026-09-23: charger, and the robot's centre when docked.
    point = approach_point(MapPose(2547, 2538), MapPose(2564, 2551), 60)
    assert point.x == pytest.approx(2594.7, abs=0.5) and point.y == pytest.approx(2574.5, abs=0.5)
    assert approach_point(None, MapPose(2564, 2551), 60) is None
    assert approach_point(MapPose(2547, 2538), None, 60) is None
    assert approach_point(MapPose(2547, 2538), MapPose(2800, 2538), 60) is None   # the dock moved


@pytest.fixture
async def pet():
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.face.enabled = False
    cfg.motion.dock_timeout_s = 0.3
    app = App(cfg, fake=True)
    await app.start()
    app.dock.poll_s = 0.02
    app.dock.start_grace_s = 0.05
    try:
        yield app
    finally:
        await app.close()


def away(pet, x=2800, y=2700):
    pet.vacuum._update(status="idle", pose=MapPose(x, y))


async def test_goes_to_the_front_of_the_dock_then_docks(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2560, 2550), MapPose(2560, 2530)
    away(pet)
    outcome, ok = await (await pet.go_home())
    assert ok and outcome == "back on my dock"
    commands = [c[0] for c in pet.vacuum.commands]
    assert commands == ["go_to", "dock"]
    assert pet.vacuum.commands[0][1:] == pytest.approx((2560, 2590))


async def test_already_in_front_of_it_docks_straight_away(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2560, 2550), MapPose(2560, 2530)
    away(pet, 2560, 2580)
    outcome, ok = await (await pet.go_home())
    assert ok and [c[0] for c in pet.vacuum.commands] == ["dock"]


async def test_a_lost_search_is_stopped_and_tried_again_from_the_front(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2560, 2550), MapPose(2560, 2530)
    away(pet)
    real_dock, tries = pet.vacuum.dock, []

    async def dock_second_time():
        tries.append(1)
        if len(tries) == 1:
            pet.vacuum.commands.append(("dock",))
            # Searching in arcs, drifting away, never arriving.
            pet.vacuum._update(status="returning", pose=MapPose(2750, 2750))
        else:
            await real_dock()
    pet.vacuum.dock = dock_second_time
    outcome, ok = await (await pet.go_home())
    assert ok
    assert [c[0] for c in pet.vacuum.commands] == ["go_to", "dock", "stop", "go_to", "dock"]


async def test_the_docked_position_is_learned_and_kept(pet):
    pet.vacuum._update(status="idle", pose=MapPose(2600, 2600))
    pet.vacuum._update(status="docked", pose=MapPose(2561, 2552), charger=MapPose(2548, 2540))
    await asyncio.sleep(0.05)
    assert pet.dock.docked_pose == MapPose(2561, 2552) and pet.dock.charger == MapPose(2548, 2540)
    assert pet.db.kv_get(DOCK_KEY) == "2561.0,2552.0,2548.0,2540.0"


async def test_docked_at_start_up_counts_too(pet):
    # The fake starts on its dock, before the dock routine is listening.
    pose = pet.vacuum.state.pose
    assert (pet.dock.docked_pose.x, pet.dock.docked_pose.y) == (pose.x, pose.y)


async def test_the_charger_jumping_to_the_docked_spot_is_not_a_move(pet):
    # 2026-09-23 12:47: leaving the dock, Valetudo moved charger_location from
    # (2548, 2540) to the robot's docked centre (2564, 2551).
    pet.vacuum._update(status="docked", pose=MapPose(2564, 2551), charger=MapPose(2548, 2540))
    await asyncio.sleep(0.05)
    pet.vacuum._update(status="idle", pose=MapPose(2643, 2605), charger=MapPose(2564, 2551))
    await asyncio.sleep(0.05)
    outcome, ok = await (await pet.go_home())
    assert ok and pet.vacuum.commands[0][0] == "go_to"
    assert pet.vacuum.commands[0][1:] == pytest.approx((2597.4, 2574.0), abs=0.5)


async def test_a_moved_dock_is_docked_from_where_the_robot_is(pet):
    pet.dock.docked_pose, pet.dock.charger = MapPose(2564, 2551), MapPose(2548, 2540)
    pet.vacuum._update(status="idle", pose=MapPose(2900, 2900), charger=MapPose(3000, 3000))
    outcome, ok = await (await pet.go_home())
    assert ok and [c[0] for c in pet.vacuum.commands] == ["dock"]


async def test_the_charger_on_top_of_the_robot_is_not_learned(pet):
    # 2026-09-23 13:30: on docking, Valetudo still had the charger at the
    # robot's centre for 2 s, which gave an approach point behind the dock.
    pet.dock.docked_pose, pet.dock.charger = MapPose(2564, 2551), MapPose(2548, 2540)
    pet.vacuum._update(status="docked", pose=MapPose(2562, 2552), charger=MapPose(2564, 2551))
    await asyncio.sleep(0.05)
    assert pet.dock.charger == MapPose(2548, 2540)          # the good pair, kept


class Replay:
    """A vacuum whose refresh() plays back a go_to's states, then keeps the last."""

    def __init__(self, *states):
        self.states = list(states)

    async def refresh(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def at(status, x, y):
    return VacuumState(reachable=True, status=status, pose=MapPose(x, y))


async def arrival(*states, target=(2562, 2601)):
    return await await_arrival(Replay(*states), "the spot", MapPose(*target), 20.0,
                               start_grace_s=0.05, poll_s=0.01)


async def test_a_go_to_that_gets_there_arrives():
    # 2026-09-26 23:41: asked for (2562, 2601), stopped 8 cm short.
    assert await arrival(at("moving", 2560, 2570), at("idle", 2561, 2593)) == ("arrived at the spot", True)


async def test_idle_short_of_the_goal_is_not_arriving():
    # 2026-09-26 23:52: a goal inside a box; round to its far side, ~40 cm off, still just "idle".
    outcome, ok = await arrival(at("moving", 2600, 2560), at("idle", 2704, 2607), target=(2690, 2570))
    assert not ok and outcome == "stopped 40 cm from the spot: something may be in the way"
    # A goal on its edge: stopped in front of it, 25 cm off.
    outcome, ok = await arrival(at("moving", 2600, 2560), at("idle", 2645, 2566), target=(2670, 2570))
    assert not ok and outcome.startswith("stopped 25 cm from the spot")


async def test_never_setting_off_depends_on_where_it_is():
    assert await arrival(at("idle", 2560, 2598)) == ("already at the spot", True)
    assert await arrival(at("idle", 2557, 2550)) == ("never set off for the spot: my base ignored me", False)


async def test_an_error_is_not_arriving():
    state = VacuumState(reachable=True, status="error", error="wheel stuck", pose=MapPose(2600, 2600))
    outcome, ok = await arrival(at("moving", 2560, 2570), state)
    assert not ok and "wheel stuck" in outcome
