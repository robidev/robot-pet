"""
Where is a person, from what the head sees? (PLAN.md 4.2, C5.) Pure functions:
a face box from the head (normalized 0..1), the head's pan and tilt, the robot's
map pose and the map give

    bearing   = pan_sign * (pan - pan_forward) + the face's offset from the image centre
    elevation = (tilt_level - tilt) / tilt_deg_per_elevation + (0.5 - cy) * vfov
    distance  = from the face's size (K_face / box height) and from the elevation
                ((eye height - camera height) / tan elevation), combined
    person    = robot + distance along heading + bearing (map cm)
    target    = the free floor standoff_m short of them, backed off toward the
                robot if that's in or near furniture (mapgeo.march_back)

Every camera constant comes from `calibration:` (scripts/calibrate_face.py fit
prints them), the camera height too: the head's design may still change it.
Until Step 2 has measured them (`camera_calibrated`), estimates say so.

Eye height decides the tilt distance: a person's own (people.face_z_m) if
known, otherwise the posture (standing; sitting or a child) whose tilt distance
agrees best with the size distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import CalibrationConfig, PersonConfig


@dataclass(frozen=True)
class PersonEstimate:
    bearing_left_deg: float                     # from the robot's heading, positive to its left
    elevation_deg: float                        # of the face, from the camera, positive up
    distance_m: float                           # horizontal, from the camera
    size_m: Optional[float]                     # the face-size distance
    tilt_m: Optional[float]                     # the elevation distance, at face_z_m
    face_z_m: float                             # the eye height used
    posture: str                                # "known" for a person's own face_z_m
    xy: Optional[tuple[float, float]] = None    # map cm, if the robot's pose is known
    target: Optional[tuple[float, float]] = None
    target_note: str = ""                       # "free", "moved back N cm", "no free floor", "no map"
    calibrated: bool = False

    def describe(self) -> str:
        side = "ahead" if abs(self.bearing_left_deg) < 5 else (
            f"{abs(self.bearing_left_deg):.0f} deg {'left' if self.bearing_left_deg > 0 else 'right'}")
        note = "" if self.calibrated else " (uncalibrated)"
        return f"~{self.distance_m:.1f} m, {side}, {self.posture}{note}"


def vfov_deg(cal: CalibrationConfig) -> float:
    if cal.vfov_deg:
        return cal.vfov_deg
    return math.degrees(2 * math.atan(math.tan(math.radians(cal.hfov_deg) / 2) * 3 / 4))


def bearing_left_deg(cx: float, pan_deg: float, cal: CalibrationConfig) -> float:
    """
    Degrees to the robot's left of its heading. The image term's sign follows
    from how a face moves as the head pans: with the head panned left, a still
    face slides to the image's right, so in an unmirrored image right is right.
    """
    axis = cal.pan_sign * (pan_deg - cal.pan_forward_deg)
    image_right_is_left = -cal.pan_sign * cal.cx_per_pan_deg_sign
    return axis + image_right_is_left * (cx - 0.5) * cal.hfov_deg


def elevation_deg(cy: float, tilt_deg: float, cal: CalibrationConfig) -> float:
    axis = (cal.tilt_level_deg - tilt_deg) / cal.tilt_deg_per_elevation_deg
    return axis + (0.5 - cy) * vfov_deg(cal)


def size_distance_m(box_h: float, cal: CalibrationConfig) -> Optional[float]:
    return cal.K_face / box_h if box_h > 0 else None


def tilt_distance_m(elevation: float, face_z_m: float, cal: CalibrationConfig,
                    pcfg: PersonConfig) -> Optional[float]:
    rise = face_z_m - cal.camera_height_m
    if elevation < pcfg.min_elevation_deg or rise <= 0:
        return None
    return rise / math.tan(math.radians(elevation))


def distance(box_h: float, elevation: float, cal: CalibrationConfig, pcfg: PersonConfig,
             face_z_m: Optional[float] = None) -> tuple[float, Optional[float], Optional[float], float, str]:
    """(distance, size distance, tilt distance, eye height, posture)."""
    by_size = size_distance_m(box_h, cal)
    if face_z_m is not None:
        options = [("known", face_z_m)]
    else:
        options = list(pcfg.postures.items())
    scored = []
    for posture, z in options:
        by_tilt = tilt_distance_m(elevation, z, cal, pcfg)
        gap = abs(by_tilt - by_size) if (by_tilt is not None and by_size is not None) else math.inf
        scored.append((gap, posture, z, by_tilt))
    gap, posture, z, by_tilt = min(scored, key=lambda s: s[0])
    if by_tilt is None:
        posture = "size only"                 # elevation too low to tell eye height from distance
    if by_tilt is not None and by_size is not None:
        d = pcfg.tilt_weight * by_tilt + (1 - pcfg.tilt_weight) * by_size
    else:
        d = by_tilt if by_tilt is not None else by_size
    if d is None:
        raise ValueError("neither a face size nor a usable elevation")
    return d, by_size, by_tilt, z, posture


def heading_unit(angle_deg: float, cal: CalibrationConfig) -> tuple[float, float]:
    """The map direction (x right, y down, cm) of a Valetudo angle (PLAN.md 4.1 calibration)."""
    theta = math.radians(cal.map_heading_sign * angle_deg + cal.map_heading_offset_deg)
    return math.cos(theta), math.sin(theta)


def project(robot_xy: tuple[float, float], robot_angle: float, bearing_left: float, distance_m: float,
            cal: CalibrationConfig) -> tuple[float, float]:
    """The person's map position (cm). Turning left grows the Valetudo angle (4.1)."""
    fx, fy = heading_unit(robot_angle, cal)
    cam = (robot_xy[0] + fx * cal.camera_forward_cm, robot_xy[1] + fy * cal.camera_forward_cm)
    ux, uy = heading_unit(robot_angle + bearing_left, cal)
    return cam[0] + ux * distance_m * 100, cam[1] + uy * distance_m * 100


def approach_target(robot_xy: tuple[float, float], person_xy: tuple[float, float], grid,
                    pcfg: PersonConfig) -> tuple[Optional[tuple[float, float]], str]:
    """The free floor standoff_m short of the person, on the line from the robot; or why not."""
    dx, dy = person_xy[0] - robot_xy[0], person_xy[1] - robot_xy[1]
    length = math.hypot(dx, dy)
    keep = max(0.0, length - pcfg.standoff_m * 100)
    goal = (robot_xy[0] + dx / length * keep, robot_xy[1] + dy / length * keep) if length else robot_xy
    if grid is None:
        return goal, "no map"
    free = grid.march_back(goal, robot_xy, pcfg.clearance_cm)
    if free is None:
        return None, "no free floor"
    moved = math.hypot(free[0] - goal[0], free[1] - goal[1])
    return free, "free" if moved < 1 else f"moved back {moved:.0f} cm"


def estimate(face, pan_deg: float, tilt_deg: float, cal: CalibrationConfig, pcfg: PersonConfig,
             robot_pose=None, grid=None, face_z_m: Optional[float] = None) -> PersonEstimate:
    """One face (io.face.Face or anything with cx, cy, h) seen at this pan and tilt."""
    bearing = bearing_left_deg(face.cx, pan_deg, cal)
    elevation = elevation_deg(face.cy, tilt_deg, cal)
    d, by_size, by_tilt, z, posture = distance(face.h, elevation, cal, pcfg, face_z_m)
    xy = target = None
    note = "robot pose unknown"
    if robot_pose is not None and getattr(robot_pose, "angle", None) is not None:
        robot_xy = (robot_pose.x, robot_pose.y)
        xy = project(robot_xy, robot_pose.angle, bearing, d, cal)
        target, note = approach_target(robot_xy, xy, grid, pcfg)
    return PersonEstimate(bearing, elevation, d, by_size, by_tilt, z, posture, xy, target, note,
                          cal.camera_calibrated)
