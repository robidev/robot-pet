import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from petd.config import CalibrationConfig, PersonConfig
from petd.io.vacuum import MapPose
from petd.spatial import person as P
from petd.spatial.mapgeo import Grid

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Box:
    cx: float
    cy: float
    h: float


def seen(cal, robot, person_xy, face_z, pan=None):
    """What the head would report for a person at person_xy (map cm), eyes face_z m up."""
    rx, ry = robot.x, robot.y
    dx, dy = (person_xy[0] - rx) / 100, (person_xy[1] - ry) / 100
    d = math.hypot(dx, dy)
    fx, fy = P.heading_unit(robot.angle, cal)
    lx, ly = P.heading_unit(robot.angle + 90, cal)            # the robot's left
    bearing = math.degrees(math.atan2(dx * lx + dy * ly, dx * fx + dy * fy))
    elev = math.degrees(math.atan2(face_z - cal.camera_height_m, d))
    if pan is None:                                           # tracking: the face centred
        pan = cal.pan_forward_deg + bearing / cal.pan_sign
        cx = 0.5
    else:                                                     # head held: the face off-centre
        off = bearing - cal.pan_sign * (pan - cal.pan_forward_deg)
        cx = 0.5 + off / (-cal.pan_sign * cal.cx_per_pan_deg_sign * cal.hfov_deg)
    tilt = cal.tilt_level_deg - elev * cal.tilt_deg_per_elevation_deg
    return Box(cx, 0.5, cal.K_face / d), pan, tilt


ROBOT = MapPose(2560.0, 2550.0, 30.0)
PCFG = PersonConfig()


@pytest.mark.parametrize("pan_sign,cx_sign", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
@pytest.mark.parametrize("where", [(2700, 2400), (2400, 2500), (2650, 2700)])
def test_a_standing_person_is_found_where_they_stand(pan_sign, cx_sign, where):
    cal = replace(CalibrationConfig(), pan_sign=pan_sign, cx_per_pan_deg_sign=cx_sign)
    for pan in (None, 95.0):                                  # tracked, and with the head held
        box, pan_deg, tilt = seen(cal, ROBOT, where, 1.55, pan)
        est = P.estimate(box, pan_deg, tilt, cal, PCFG, ROBOT)
        assert est.posture == "standing"
        assert math.dist(est.xy, where) < 3, (est, where)


def test_a_sitting_person_or_a_child_is_told_from_a_standing_one():
    cal = CalibrationConfig()
    box, pan, tilt = seen(cal, ROBOT, (2700, 2450), 1.2)
    est = P.estimate(box, pan, tilt, cal, PCFG, ROBOT)
    assert est.posture == "sitting or a child"
    assert math.dist(est.xy, (2700, 2450)) < 3


def test_a_known_eye_height_is_used_and_a_changed_camera_height_counts():
    cal = replace(CalibrationConfig(), camera_height_m=0.35)   # the head raised on a new mount
    box, pan, tilt = seen(cal, ROBOT, (2500, 2350), 1.40)
    est = P.estimate(box, pan, tilt, cal, PCFG, ROBOT, face_z_m=1.40)
    assert est.posture == "known" and math.dist(est.xy, (2500, 2350)) < 3
    assert abs(est.tilt_m - est.size_m) < 0.01


def test_too_little_elevation_leaves_the_size_alone():
    cal = CalibrationConfig()
    box, pan, tilt = seen(cal, ROBOT, (3200, 2550), 0.5)       # low and far: ~2 deg up
    est = P.estimate(box, pan, tilt, cal, PCFG, ROBOT)
    assert est.tilt_m is None and est.posture == "size only"
    assert abs(est.distance_m - est.size_m) < 1e-9


def test_without_a_pose_there_is_a_distance_but_no_place():
    cal = CalibrationConfig()
    box, pan, tilt = seen(cal, ROBOT, (2700, 2400), 1.55)
    est = P.estimate(box, pan, tilt, cal, PCFG, MapPose(2560, 2550, None))
    assert est.xy is None and est.distance_m > 1 and not est.calibrated
    assert "uncalibrated" in est.describe()


def test_the_target_is_short_of_them_and_off_the_furniture():
    grid = Grid.from_valetudo(json.loads((FIXTURES / "valetudo_map_2026-09-26.json").read_text()))
    robot = (2560.0, 2600.0)
    def short_of(p):                                          # 1 m short, toward the robot
        d = math.dist(robot, p)
        return (robot[0] + (p[0] - robot[0]) * (d - 100) / d, robot[1] + (p[1] - robot[1]) * (d - 100) / d)
    open_floor = next((x, y) for x in range(2300, 2900, 10) for y in range(2300, 2900, 10)
                      if 200 <= math.dist(robot, (x, y)) <= 300 and grid.is_free(*short_of((x, y)), 30))
    target, note = P.approach_target(robot, open_floor, grid, PCFG)
    assert note == "free"
    assert abs(math.dist(robot, target) - (math.dist(robot, open_floor) - 100)) < 1
    # Closer than the standoff: it stays where it is.
    near = short_of(open_floor)
    assert P.approach_target(robot, ((robot[0] + near[0]) / 2, (robot[1] + near[1]) / 2), grid, PCFG)[0] == robot
    # A person in a wall or behind furniture: the target backs off toward the robot.
    walled = next((x, y) for x in range(2600, 3400, 10) for y in range(2400, 2800, 10)
                  if not grid.is_floor(x, y) and math.dist(robot, (x, y)) > 250)
    target, note = P.approach_target(robot, walled, grid, PCFG)
    assert target is None or grid.is_free(*target, PCFG.clearance_cm)
    assert note in ("free", "no free floor") or note.startswith("moved back")
