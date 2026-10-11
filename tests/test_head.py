"""
The head's directions as measured on the robot (spatial/head.py; PLAN.md,
"Head and eye conventions"). Each test is an observation: if one fails after
a change, the change has the sign wrong, not the observation. After a
remount, measure again and update the calibration, not these.
"""

import asyncio

import pytest

from petd.brain.expressions import EMOTIONS, GLANCES, Expressions
from petd.brain.tools import _describe_look, _describe_position
from petd.bus import EventBus
from petd.config import CalibrationConfig, Config
from petd.io.face import FakeFace
from petd.spatial.head import eye_xy, head_direction, head_servo, image_left
from petd.spatial.person import bearing_left_deg, elevation_deg

# The calibration of 2026-10-11, after the camera was remounted.
TODAY = CalibrationConfig(hfov_deg=83.7, pan_forward_deg=76.4, pan_sign=-1.0, cx_per_pan_deg_sign=-1.0,
                          tilt_level_deg=82.3, tilt_deg_per_elevation_deg=-1.101)


@pytest.mark.parametrize("cal", [TODAY, CalibrationConfig()])
def test_servo_angles_and_directions_convert_both_ways(cal):
    for left, up in ((0, 0), (30, 0), (-45, 10), (60, -12)):
        pan, tilt = head_servo(left, up, cal)
        assert head_direction(pan, tilt, cal) == pytest.approx((left, up))


def test_the_pan_servo_grows_to_the_robots_right():
    # 2026-10-05: pan 75 -> 100 turned the head to the robot's right (seen).
    assert head_direction(100.0, TODAY.tilt_level_deg, TODAY)[0] < 0
    # 2026-10-11: a face 30 deg to the robot's left was centred at pan 45.5.
    assert head_direction(45.5, TODAY.tilt_level_deg, TODAY)[0] == pytest.approx(30.9)
    assert head_servo(-30, 0, TODAY)[0] == pytest.approx(106.4)      # "30 degrees right"


def test_higher_tilt_looks_up():
    # 2026-10-05: tilt 110 -> 95 looked down (more floor in the snapshot).
    assert head_direction(TODAY.pan_forward_deg, 95.0, TODAY)[1] < head_direction(TODAY.pan_forward_deg, 110.0, TODAY)[1]
    assert head_servo(0, 20, TODAY)[1] == pytest.approx(82.3 + 22.02)     # 1.1 servo degrees a degree


def test_the_image_right_is_the_robots_right():
    # With the head panned left a still face slides to the image's right
    # (cx_per_pan_deg_sign), so the image isn't mirrored.
    assert image_left(0.9, TODAY) < 0 < image_left(0.1, TODAY)
    assert _describe_position(0.9, TODAY) == "to my right"
    straight = TODAY.pan_forward_deg
    assert bearing_left_deg(0.9, straight, TODAY) < 0 < bearing_left_deg(0.1, straight, TODAY)
    assert elevation_deg(0.1, TODAY.tilt_level_deg, TODAY) > 0 > elevation_deg(0.9, TODAY.tilt_level_deg, TODAY)


def test_the_eye_x_grows_to_the_robots_left_and_y_downward():
    # 2026-10-11, seen on the display: x +1 sat at the robot's left, y +1 at the bottom.
    assert eye_xy(1.0, 0.0) == (1.0, 0.0)
    assert eye_xy(0.0, 1.0) == (0.0, -1.0)


async def test_glances_and_emotes_look_the_way_they_say():
    face = FakeFace(Config().face, EventBus())
    expressions = Expressions(face, TODAY)
    eye = lambda: next(c for c in reversed(face.commands) if c[0] == "eye")      # noqa: E731
    for name, want_x, want_y in (("left", 1, 0), ("right", -1, 0), ("up", 0, -1), ("down", 0, 1)):
        await expressions.glance(name)
        _, x, y, _ = eye()
        assert (x > 0) - (x < 0) == want_x and (y > 0) - (y < 0) == want_y, name
        await asyncio.sleep(0)
    await expressions.emote("thinking")                    # up and to the left
    _, x, y, _ = eye()
    assert x > 0 and y < 0
    await expressions.emote("sleepy")                      # drooping
    assert eye()[2] > 0
    assert EMOTIONS["thinking"][:2] == (0.4, 0.35) and GLANCES["left"][0] > 0


def test_where_the_head_looks_is_said_in_the_robots_terms():
    assert _describe_look(106.4, 82.3, TODAY) == "30° right, level"
    assert _describe_look(46.4, 104.3, TODAY) == "30° left, 20° up"
