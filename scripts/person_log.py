"""
Where petd puts a person, recorded under a label (PLAN.md Step 4, C5).

    .venv/bin/python scripts/person_log.py "2.0m 30 left standing"

With petd running, reads GET /person ten times over ~5 s with one person in
view and appends the medians to runtime/calibration/person-log.jsonl (or
--out): distance and bearing, both distances (face size, tilt), the posture,
the map position and the approach target, with the robot's pose. --map also
saves the dashboard map beside it (LABEL as a file name).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "calibration" / "person-log.jsonl"


def get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read()
    except OSError as e:
        sys.exit(f"petd isn't answering at {url}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label")
    parser.add_argument("--petd", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--map", action="store_true", help="also save the dashboard's map.png")
    args = parser.parse_args()

    rows = []
    for _ in range(10):
        people = json.loads(get(f"{args.petd}/person")).get("people", [])
        if len(people) == 1:
            rows.append(people[0])
        time.sleep(0.5)
    pose = json.loads(get(f"{args.petd}/status"))["vacuum"]["pose"]
    if not rows:
        sys.exit(f"{args.label}: no single person in view")

    def med(key):
        values = [p[key] for p in rows if p.get(key) is not None]
        return round(statistics.median(values), 3) if values else None

    def med_xy(key):
        points = [p[key] for p in rows if p.get(key)]
        return [round(statistics.median(v[i] for v in points)) for i in (0, 1)] if points else None

    out = {"label": args.label, "t": time.time(), "samples": len(rows), "robot_pose": pose,
           **{k: med(k) for k in ("bearing_left_deg", "elevation_deg", "distance_m", "size_m", "tilt_m", "face_z_m")},
           "posture": statistics.mode(p["posture"] for p in rows),
           "xy": med_xy("xy"), "target": med_xy("target"),
           "targets_found": sum(1 for p in rows if p.get("target")),
           "target_note": statistics.mode(str(p.get("target_note")) for p in rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out))
    if args.map:
        png = args.out.parent / (re.sub(r"[^A-Za-z0-9.]+", "-", args.label).strip("-") + ".png")
        png.write_bytes(get(f"{args.petd}/map.png"))
        print(png)


if __name__ == "__main__":
    main()
