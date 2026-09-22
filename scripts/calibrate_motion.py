"""
What does Valetudo's manual-control vector do to the wheels? (PLAN.md C3/C4)

HighResolutionManualControl takes {velocity: -1..1, angle: -180..180}, and
nothing documents the units. Player's odometry streams at 50 Hz while
manual control is armed (and not otherwise), so this sends a series of
short, slow commands and measures each one: forward speed, turn rate,
and whether "angle" turns the robot in place or steers it.

It also reads Valetudo's map pose before and after a turn, which pins
down the map's angle convention (the sign, relative to odometry's CCW).

    .venv/bin/python scripts/calibrate_motion.py            # needs ~1 m clear all round

THE ROBOT MOVES: each step is at most 1.5 s at a low speed, it stops on
a stall, and Ctrl-C disarms manual control. Do not run it docked.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "playerc-client"))
sys.path.insert(0, str(ROOT / "vacuum-api"))

import playerc_client as pc                 # noqa: E402
from valetudo_client import ValetudoClient  # noqa: E402

RESEND_S = 0.15

# (label, velocity, angle, seconds). Turns alternate direction, drives
# come back, so the robot ends up roughly where it started.
STEPS = [
    ("angle +30, no velocity", 0.0, 30.0, 1.5),
    ("angle -30, no velocity", 0.0, -30.0, 1.5),
    ("angle +90, no velocity", 0.0, 90.0, 1.5),
    ("angle -90, no velocity", 0.0, -90.0, 1.5),
    ("forward 0.1", 0.1, 0.0, 1.5),
    ("backward 0.1", -0.1, 0.0, 1.5),
    ("forward 0.25", 0.25, 0.0, 1.5),
    ("backward 0.25", -0.25, 0.0, 1.5),
    ("forward 0.2, angle +30", 0.2, 30.0, 1.5),
    ("backward 0.2, angle +30", -0.2, 30.0, 1.5),
    # Not undone: the net turn that the map's angle convention is read from.
    ("final turn, angle +90", 0.0, 90.0, 2.0),
]
# If "angle" turns nothing at zero velocity, the final turn is done steering.
FALLBACK_TURN = ("final turn, forward 0.15 angle +90", 0.15, 90.0, 2.0)


class Odometry:
    """Player position2d on a thread; `latest` and a sample log."""

    def __init__(self, host: str):
        self.client = pc.PlayerClient(host, timeout=60)
        self.client.connect()
        self.client.subscribe(pc.PLAYER_POSITION2D_CODE, 0)
        self.samples: list = []
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        for pose in self.client.read_position2d():
            with self.lock:
                self.samples.append(pose)

    @property
    def latest(self):
        with self.lock:
            return self.samples[-1] if self.samples else None

    def since(self, index: int) -> list:
        with self.lock:
            return self.samples[index:]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def valetudo_pose(v: ValetudoClient):
    p = v.get_position()
    return None if p is None else (p.x, p.y, p.angle)


def run_step(v: ValetudoClient, odo: Odometry, velocity: float, angle: float, seconds: float) -> dict:
    start = len(odo.samples)
    t_end = time.monotonic() + seconds
    stalled = False
    while time.monotonic() < t_end:
        v.drive_vector(velocity, angle)
        time.sleep(RESEND_S)
        last = odo.latest
        if last is not None and last.stalled:
            stalled = True
            break
    v.drive_vector(0.0, 0.0)
    time.sleep(1.0)                          # let it coast to a stop, and measure that too
    samples = odo.since(start)
    if len(samples) < 2:
        return {"error": "no odometry"}
    first, last = samples[0], samples[-1]
    yaw = sum(wrap(b.yaw - a.yaw) for a, b in zip(samples, samples[1:]))
    # Steady part: the middle half of the commanded window.
    t0 = first.timestamp
    steady = [s for s in samples if t0 + seconds * 0.25 <= s.timestamp <= t0 + seconds * 0.75]
    return {
        "dx": last.x - first.x, "dy": last.y - first.y,
        "dist": math.hypot(last.x - first.x, last.y - first.y),
        "dyaw_deg": math.degrees(yaw),
        "v": sum(s.vx for s in steady) / len(steady) if steady else float("nan"),
        "w_deg": math.degrees(sum(s.vyaw for s in steady) / len(steady)) if steady else float("nan"),
        "stalled": stalled or any(s.stalled for s in samples),
        "heading_before": math.degrees(first.yaw),
    }


def main(args) -> None:
    v = ValetudoClient(args.host)
    status = v.get_status().get("value")
    if status == "docked":
        sys.exit("The robot is docked. Undock it (and give it ~1 m of room) first.")
    print(f"robot is {status}; battery {v.get_battery().get('level')}%")
    before = valetudo_pose(v)
    print(f"Valetudo pose before: {before}")
    if not args.yes and input("The robot will move in short, slow bursts. Clear around it? [y/N] ").lower() != "y":
        return

    odo = Odometry(args.host)
    v.enable_manual_control()
    try:
        deadline = time.monotonic() + 3
        while odo.latest is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if odo.latest is None:
            sys.exit("Player sent no odometry with manual control armed.")
        rows = []
        for label, velocity, angle, seconds in STEPS:
            r = run_step(v, odo, velocity, angle, seconds)
            rows.append((label, velocity, angle, seconds, r))
            if "error" in r:
                print(f"{label}: {r['error']}")
                break
            print(f"{label:26s} moved {r['dist']*100:5.1f} cm (odom dx {r['dx']*100:+5.1f}, dy {r['dy']*100:+5.1f}), "
                  f"turned {r['dyaw_deg']:+6.1f} deg | steady v {r['v']*100:+5.1f} cm/s, "
                  f"w {r['w_deg']:+6.1f} deg/s{'  STALLED' if r['stalled'] else ''}")
            if r["stalled"]:
                print("stall/bump detected; stopping here.")
                break
        spun = [r for label, velocity, *_, r in rows if velocity == 0.0 and "error" not in r]
        if spun and not r.get("stalled") and max(abs(x["dyaw_deg"]) for x in spun) < 5:
            label, velocity, angle, seconds = FALLBACK_TURN
            r = run_step(v, odo, velocity, angle, seconds)
            rows.append((label, velocity, angle, seconds, r))
            print(f"{label:26s} moved {r['dist']*100:5.1f} cm, turned {r['dyaw_deg']:+6.1f} deg")
    finally:
        try:
            v.drive_vector(0.0, 0.0)
        finally:
            v.disable_manual_control()

    # The map angle convention: turn by a known amount and read Valetudo again.
    time.sleep(3)
    after = valetudo_pose(v)
    print(f"Valetudo pose after:  {after}")
    turned = sum(r["dyaw_deg"] for *_, r in rows if "error" not in r)
    if before and after and before[2] is not None and after[2] is not None:
        d_map = (after[2] - before[2] + 180) % 360 - 180
        print(f"net odometry turn {turned:+.1f} deg (CCW positive); Valetudo angle changed {d_map:+.1f} deg")
        print("  -> map_angle_sign = %s" % ("+1 (same as odometry)" if d_map * turned > 0
                                             else "-1 (Valetudo angles run clockwise)"
                                             if abs(turned) > 10 else "unclear: net turn too small"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.101.43")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    main(parser.parse_args())
