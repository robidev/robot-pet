import asyncio

import pytest

from petd.app import App
from petd.config import Config
from petd.io.vacuum import MapPose
from petd.spatial.dock import DOCKED_POSE_KEY, approach_point


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
    pet.dock.docked_pose = MapPose(2560, 2550)       # fake charger is at (2560, 2530)
    away(pet)
    outcome, ok = await (await pet.go_home())
    assert ok and outcome == "back on my dock"
    commands = [c[0] for c in pet.vacuum.commands]
    assert commands == ["go_to", "dock"]
    assert pet.vacuum.commands[0][1:] == pytest.approx((2560, 2590))


async def test_already_in_front_of_it_docks_straight_away(pet):
    pet.dock.docked_pose = MapPose(2560, 2550)
    away(pet, 2560, 2580)
    outcome, ok = await (await pet.go_home())
    assert ok and [c[0] for c in pet.vacuum.commands] == ["dock"]


async def test_a_lost_search_is_stopped_and_tried_again_from_the_front(pet):
    pet.dock.docked_pose = MapPose(2560, 2550)
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
    pet.vacuum._update(status="docked", pose=MapPose(2561, 2552))
    await asyncio.sleep(0.05)
    assert pet.dock.docked_pose == MapPose(2561, 2552)
    assert pet.db.kv_get(DOCKED_POSE_KEY) == "2561.0,2552.0"


async def test_docked_at_start_up_counts_too(pet):
    # The fake starts on its dock, before the dock routine is listening.
    pose = pet.vacuum.state.pose
    assert (pet.dock.docked_pose.x, pet.dock.docked_pose.y) == (pose.x, pose.y)
