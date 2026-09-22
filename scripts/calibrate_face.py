"""
Face-distance calibration (PLAN.md 4.2, C3): where is a person, from what
the head sees?

    .venv/bin/python scripts/calibrate_face.py aim                   # head up, tracking on
    .venv/bin/python scripts/calibrate_face.py capture 2.0 [--bearing 45]
    .venv/bin/python scripts/calibrate_face.py hfov                  # stand still at ~2 m
    .venv/bin/python scripts/calibrate_face.py fit --face-height 1.62

`capture` records ~4 s of GET /api/face/current with tracking on (one face,
centred by the servos) at a measured distance in metres from the camera,
straight ahead unless --bearing says how many degrees to the robot's left
(negative: right). `hfov` sweeps the pan servo by hand and watches the face
move across the frame, which gives the camera's horizontal field of view.
`fit` turns runtime/calibration/face.jsonl into the `calibration:` values.

Conventions checked so far: tilt 90 is level and LOWER tilt looks UP
(snapshots at 60 and 120). Nothing moves but the head.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "face-api"))

from face_client import FaceApiClient  # noqa: E402

LOG = ROOT / "runtime" / "calibration" / "face.jsonl"
CAMERA_HEIGHT_M = 0.20


def client(args) -> FaceApiClient:
    return FaceApiClient(args.host, timeout=5)


def sample(c: FaceApiClient, seconds: float) -> list[dict]:
    """Frames with exactly one face, as flat dicts."""
    rows, seen = [], set()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        cur = c.get_current_faces()
        key = (cur.utc, len(cur.faces))
        if key not in seen and len(cur.faces) == 1:
            seen.add(key)
            f = cur.faces[0]
            box = f["box"]
            rows.append({"pan": cur.servo.pan_deg, "tilt": cur.servo.tilt_deg,
                         "cx": (box["left"] + box["right"]) / 2, "cy": (box["top"] + box["bottom"]) / 2,
                         "w": box["right"] - box["left"], "h": box["bottom"] - box["top"],
                         "id": f.get("id", -1)})
        time.sleep(0.15)
    return rows


def medians(rows: list[dict]) -> dict:
    return {k: round(statistics.median(r[k] for r in rows), 4) for k in ("pan", "tilt", "cx", "cy", "w", "h")}


def cmd_aim(args) -> None:
    c = client(args)
    c.set_servo(mode="manual", pan_deg=90, tilt_deg=args.tilt)
    time.sleep(1.0)
    c.set_servo(mode="track")
    print(f"head at pan 90, tilt {args.tilt}; tracking on")


def cmd_capture(args) -> None:
    c = client(args)
    if args.tilt is not None:
        # Fixed head, no tracking: safe near the tilt stop, and where the face
        # lands in the frame then measures the vertical field of view.
        c.set_servo(mode="manual", pan_deg=90, tilt_deg=args.tilt)
    else:
        c.set_servo(mode="track")
    print("settling 2 s ...")
    time.sleep(2.0)
    rows = sample(c, args.seconds)
    if len(rows) < 3:
        sys.exit(f"only {len(rows)} frames with exactly one face; is the face in view?")
    m = medians(rows)
    spread = round(statistics.pstdev(r["h"] for r in rows), 4)
    entry = {"kind": "capture", "distance_m": args.distance, "bearing_deg": args.bearing,
             "fixed_tilt": args.tilt,
             "frames": len(rows), "h_spread": spread, **m, "t": time.time()}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry))


def cmd_hfov(args) -> None:
    c = client(args)
    tilt = args.tilt            # held fixed: no tracking near the tilt stop
    points = []
    for pan in (80, 90, 100):
        c.set_servo(mode="manual", pan_deg=pan, tilt_deg=tilt)
        time.sleep(1.5)
        rows = sample(c, 2.0)
        if rows:
            points.append((pan, statistics.median(r["cx"] for r in rows)))
            print(f"pan {pan}: face cx {points[-1][1]:.3f} ({len(rows)} frames)")
        else:
            print(f"pan {pan}: face not in view")
    c.set_servo(mode="manual", pan_deg=90, tilt_deg=tilt)
    if len(points) >= 2:
        (p0, x0), (p1, x1) = points[0], points[-1]
        slope = (x1 - x0) / (p1 - p0)            # image widths per pan degree
        entry = {"kind": "hfov", "points": points, "cx_per_pan_deg": slope,
                 "hfov_deg": abs(1 / slope) if slope else None, "t": time.time()}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry))


def cmd_fit(args) -> None:
    entries = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    ahead = [e for e in entries if e["kind"] == "capture" and abs(e.get("bearing_deg", 0)) < 1]
    side = [e for e in entries if e["kind"] == "capture" and abs(e.get("bearing_deg", 0)) >= 1]
    hfov = [e for e in entries if e["kind"] == "hfov" and e.get("hfov_deg")]
    out: dict = {}

    if ahead:
        ks = [e["distance_m"] * e["h"] for e in ahead]
        out["K_face"] = round(statistics.median(ks), 4)
        print("distance x box height:", [round(k, 3) for k in ks], "-> K_face", out["K_face"])
        pan0 = statistics.median(e["pan"] for e in ahead)
        out["pan_forward_deg"] = round(pan0, 1)

    if hfov:
        out["hfov_deg"] = round(statistics.median(e["hfov_deg"] for e in hfov), 1)
        # Positive: the face moves right in the image as pan grows.
        out["cx_per_pan_deg_sign"] = 1 if statistics.median(e["cx_per_pan_deg"] for e in hfov) > 0 else -1

    if side and ahead:
        e = side[0]
        out["pan_sign"] = 1 if (e["pan"] - out["pan_forward_deg"]) * e["bearing_deg"] > 0 else -1
        print(f"bearing {e['bearing_deg']:+} deg -> pan {e['pan']} (forward {out['pan_forward_deg']}):",
              "pan grows to the left" if out["pan_sign"] > 0 else "pan grows to the right")

    fixed = [e for e in ahead if e.get("fixed_tilt") is not None]
    if args.face_height and len(fixed) >= 2:
        # Head held still: elevation = centre_elevation + (0.5 - cy) * vfov.
        elev = [math.degrees(math.atan2(args.face_height - CAMERA_HEIGHT_M, e["distance_m"])) for e in fixed]
        off = [0.5 - e["cy"] for e in fixed]
        n = len(fixed)
        mx, my = sum(off) / n, sum(elev) / n
        sxx = sum((x - mx) ** 2 for x in off)
        vfov = sum((x - mx) * (y - my) for x, y in zip(off, elev)) / sxx if sxx else float("nan")
        centre = my - vfov * mx
        tilt = fixed[0]["fixed_tilt"]
        out["vfov_deg"] = round(vfov, 1)
        # Lower tilt looks up; assuming servo degrees are real degrees.
        out["tilt_level_deg"] = round(tilt + centre, 1)
        print(f"fixed tilt {tilt}: frame centre looks {centre:.1f} deg up, vertical FOV {vfov:.1f} deg")
        for x, y, e in zip(off, elev, fixed):
            print(f"  {e['distance_m']} m: elevation {y:.1f} deg, cy {e['cy']:.3f} (model {centre + vfov * x:.1f})")

    tracked = [e for e in ahead if e.get("fixed_tilt") is None]
    if args.face_height and len(tracked) >= 2:
        ahead = tracked
        # tilt = level - elevation * scale (lower tilt looks up), with the face
        # nearly centred by tracking; the residual cy offset is ignored here.
        elev = [math.degrees(math.atan2(args.face_height - CAMERA_HEIGHT_M, e["distance_m"])) for e in ahead]
        tilts = [e["tilt"] for e in ahead]
        n = len(elev)
        mx, my = sum(elev) / n, sum(tilts) / n
        sxx = sum((x - mx) ** 2 for x in elev)
        scale = -sum((x - mx) * (y - my) for x, y in zip(elev, tilts)) / sxx if sxx else 1.0
        level = my + scale * mx
        out["tilt_level_deg"] = round(level, 1)
        out["tilt_deg_per_elevation_deg"] = round(scale, 3)
        for x, y, e in zip(elev, tilts, ahead):
            print(f"  {e['distance_m']} m: elevation {x:.1f} deg, tilt {y:.1f} (model {level - scale * x:.1f}),"
                  f" cy {e['cy']:.2f}")
    print("\ncalibration:")
    for k, v in out.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.101.40")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("aim"); p.add_argument("--tilt", type=float, default=60.0); p.set_defaults(fn=cmd_aim)
    p = sub.add_parser("capture"); p.add_argument("distance", type=float)
    p.add_argument("--bearing", type=float, default=0.0); p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--tilt", type=float, help="hold the head at this tilt instead of tracking")
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("hfov"); p.add_argument("--tilt", type=float, default=60.0); p.set_defaults(fn=cmd_hfov)
    p = sub.add_parser("fit"); p.add_argument("--face-height", type=float); p.set_defaults(fn=cmd_fit)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
