"""
The head's directions, in one place. Its signs have gone wrong four times
(2026-09-23 tilt, 2026-10-05 the remount, 2026-10-11 the brain's own servo
arithmetic, 2026-10-11 the eye), so every conversion between the robot's
terms and the device's lives here, and nothing else does its own sign
arithmetic.

The robot's terms, used everywhere above the devices (the brain, behaviours,
person geometry, logs it reads):
- left is the robot's own left, positive; right is negative.
- up is above level, positive; down is negative.
- the body turns counter-clockwise (to its left) for positive degrees
  (spatial/motion.py turn_by).

The devices, and how each maps (PLAN.md, "Head and eye conventions"):
- Servo angles (pan, tilt; 0-180): the calibration's pan_forward_deg,
  pan_sign, tilt_level_deg and tilt_deg_per_elevation_deg, measured with
  scripts/calibrate_face.py after every remount. Since 2026-10-05 pan grows
  to the robot's right and higher tilt looks up. Only this module, the
  servo limits (face.tilt_min_deg/tilt_max_deg, search_pan_min/max_deg) and
  the device adapters see servo angles.
- The camera image (cx, cy; 0..1): upright and unmirrored as the firmware
  sets it up (hmirror and vflip both on, for the sensor mounted upside
  down): cx grows to the robot's right, cy grows downward. The calibration's
  cx_per_pan_deg_sign records the horizontal measurement.
- The eye on the display (x, y; -1..1): x grows to the robot's left, y grows
  downward, seen on the display on 2026-10-11 (x +1 at the robot's left, y
  +1 at the bottom). The firmware's own face-following gaze agrees; petd's
  glances and emotes had both axes the wrong way round until then.
"""

from __future__ import annotations

from ..config import CalibrationConfig

# The eye's device axes in the robot's terms: x +1 is the robot's left, y +1
# is down (seen on the display, 2026-10-11).
EYE_X_PER_LEFT = 1.0
EYE_Y_PER_UP = -1.0


def head_offset(left_deg: float, up_deg: float, cal: CalibrationConfig) -> tuple[float, float]:
    """Degrees to the robot's left and up, as pan and tilt offsets (tilt 1.1 a degree since 2026-10-11)."""
    return cal.pan_sign * left_deg, -up_deg * cal.tilt_deg_per_elevation_deg


def head_servo(left_deg: float, up_deg: float, cal: CalibrationConfig) -> tuple[float, float]:
    """Servo angles (pan, tilt) for degrees to the robot's left of straight ahead and up from level."""
    pan_offset, tilt_offset = head_offset(left_deg, up_deg, cal)
    return cal.pan_forward_deg + pan_offset, cal.tilt_level_deg + tilt_offset


def head_direction(pan_deg: float, tilt_deg: float, cal: CalibrationConfig) -> tuple[float, float]:
    """Where servo angles point: degrees to the robot's left of straight ahead, and up from level."""
    return (cal.pan_sign * (pan_deg - cal.pan_forward_deg),
            (cal.tilt_level_deg - tilt_deg) / cal.tilt_deg_per_elevation_deg)


def image_left(cx: float, cal: CalibrationConfig) -> float:
    """
    How far a point sits to the robot's left of the image centre, in image
    widths (-0.5..0.5). With the head panned left a still face slides to the
    image's right, so in this unmirrored image right is the robot's right.
    """
    return -cal.pan_sign * cal.cx_per_pan_deg_sign * (cx - 0.5)


def image_up(cy: float) -> float:
    """How far a point sits above the image centre, in image heights (-0.5..0.5)."""
    return 0.5 - cy


def eye_xy(left: float, up: float) -> tuple[float, float]:
    """The eye's device position for a gaze to the robot's left and up (each -1..1)."""
    return EYE_X_PER_LEFT * left, EYE_Y_PER_UP * up
