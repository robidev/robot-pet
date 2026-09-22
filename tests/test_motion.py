import asyncio

import pytest

from petd.bus import EventBus
from petd.config import MotionConfig
from petd.io.vacuum import FakeVacuum
from petd.spatial.motion import Motion


def make(warmup_s=0.0, **cfg):
    vacuum = FakeVacuum(EventBus(), warmup_s=warmup_s)
    config = MotionConfig(warmup_s=warmup_s, resend_s=0.02, settle_s=0.4, idle_disarm_s=0.3,
                          **cfg)
    return vacuum, Motion(vacuum, vacuum.odometry, config)


@pytest.mark.parametrize("degrees", [60, -30, 12])
async def test_turns_land_within_tolerance(degrees):
    vacuum, motion = make()
    result = await motion.turn_by(degrees)
    assert result.ok, result
    assert abs(result.done - degrees) <= 4
    # Valetudo's angle is clockwise: a left (positive) turn sends negative angles.
    turning = [a for c, v, a in (x for x in vacuum.commands if x[0] == "manual_move") if a]
    assert turning and all((a < 0) == (degrees > 0) for a in turning[:3])
    await motion.close()


@pytest.mark.parametrize("cm", [15, -8])
async def test_moves_land_within_tolerance(cm):
    vacuum, motion = make()
    result = await motion.move_by(cm)
    assert result.ok, result
    assert abs(result.done - cm) <= 3
    speeds = [v for c, v, a in (x for x in vacuum.commands if x[0] == "manual_move") if v]
    assert all(abs(v) < 0.3 for v in speeds)       # the V1 ignores >= 0.3
    await motion.close()


async def test_requests_are_clamped():
    vacuum, motion = make(max_turn_deg=20)
    assert (await motion.turn_by(400)).asked == 20
    await motion.close()


async def test_one_warmup_for_consecutive_motions_then_disarm_when_idle():
    vacuum, motion = make(warmup_s=0.2)
    await motion.turn_by(30)
    await motion.move_by(10)
    assert vacuum.commands.count(("manual_start",)) == 1
    # Moves sent before the lidar was up would have been ignored; none were.
    assert motion.armed
    await asyncio.sleep(0.6)
    assert not motion.armed and vacuum.commands[-1] == ("manual_end",)


async def test_a_stall_ends_the_motion():
    vacuum, motion = make()

    async def block_soon():
        await asyncio.sleep(0.1)
        vacuum.odometry.stalled = True
    asyncio.create_task(block_soon())
    result = await motion.move_by(40)
    assert not result.ok and "bumped" in result.reason and result.done < 40
    await motion.close()


async def test_cancelling_sends_a_stop_vector():
    vacuum, motion = make()
    task = asyncio.create_task(motion.turn_by(180))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert vacuum.commands[-1][:3] == ("manual_move", 0.0, 0.0)
    await motion.disarm()
    assert vacuum.commands[-1] == ("manual_end",)
