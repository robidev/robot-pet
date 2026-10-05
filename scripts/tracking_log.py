"""
Why does face tracking oscillate? Records what the
head's tracker saw and did, one row per detection pass, and tells overshoot
(the loop over-corrects) apart from jitter (the box wobbles past the deadband).

    curl "http://192.168.101.40/api/servo?mode=track"   # tracking on, yourself
    .venv/bin/python scripts/tracking_log.py record --seconds 20 --label close-0.6m
    .venv/bin/python scripts/tracking_log.py analyze runtime/calibration/tracking-*.jsonl

Read-only: it only polls GET /api/face/current and /api/status. Each pass's
servo pose is the one its frame was captured at, so a pass's error and the
tilt at the next pass show what the tracker's correction actually did.

The summary, for tilt and for pan, per face size (box height as a fraction
of the frame):

- step ratio: how much of the error one correction removed, measured as the
  next pass's error over this one's, for passes the tracker acted on.
  ~0 = landed on the face; negative = crossed the centre (overshoot);
  below -1 = each swing bigger than the last.
- still jitter: how much the face centre moves between passes while the tilt
  stayed put, against the tracker's deadband (0.06). Jitter near or past the
  deadband makes the tracker chase noise.

And whether the image matches the pose reported with it: how well the face's
vertical position is explained by the tilt of the same pass, or of the
previous one. Before 2026-09-23 the previous pass won (R^2 0.80 against 0.44 at 0.6 m):
the pose was sampled when a frame was copied, not when it was taken, so the
tracker corrected from a pose the head had already left, and oscillated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "face-api"))

from face_client import FaceApiClient  # noqa: E402

sys.path.insert(0, str(ROOT))
from petd.config import load_config  # noqa: E402
_FACE = load_config().face
TILT_MIN, TILT_MAX = _FACE.tilt_min_deg, _FACE.tilt_max_deg     # the mount's stops

LOG_DIR = ROOT / "runtime" / "calibration"
DEADBAND = 0.06              # servo_service.cpp tracking_deadband
STILL_DEG = 0.5              # tilt moves smaller than this count as "didn't move"
SIZE_BUCKETS = (0.0, 0.15, 0.25, 0.35, 1.01)


def record(args) -> None:
    c = FaceApiClient(args.host, timeout=5)
    servo = c.get_status()["servo"]
    if servo.get("mode") != "track":
        print(f"warning: servo mode is {servo.get('mode')!r}, not 'track'; "
              "turn tracking on first: /api/servo?mode=track")
    print(f"gains: pan {servo.get('tracking_gain')}, tilt {servo.get('tracking_tilt_gain')}, "
          f"rate {servo.get('tracking_rate')} deg/s. Recording {args.seconds:.0f} s...")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"tracking-{time.strftime('%Y%m%d-%H%M%S')}-{args.label}.jsonl"
    rows, last_seq = [], None
    end = time.monotonic() + args.seconds
    with path.open("w") as out:
        out.write(json.dumps({"meta": True, "label": args.label, "servo": servo}) + "\n")
        while time.monotonic() < end:
            cur = c.get_current_faces()
            if cur.frame_seq != last_seq:
                last_seq = cur.frame_seq
                row = {"seq": cur.frame_seq, "t": time.time(), "age_ms": cur.age_ms,
                       "pan": cur.servo.pan_deg, "tilt": cur.servo.tilt_deg, "faces": len(cur.faces),
                       **cur.timing}
                if cur.faces:
                    box = cur.faces[0]["box"]
                    row.update(cx=(box["left"] + box["right"]) / 2, cy=(box["top"] + box["bottom"]) / 2,
                               h=box["bottom"] - box["top"])
                rows.append(row)
                out.write(json.dumps(row) + "\n")
            time.sleep(0.1)
    print(f"{len(rows)} passes -> {path.relative_to(ROOT)}")
    summarize(rows, servo)


def load(paths: list[str]) -> None:
    for name in paths:
        lines = [json.loads(line) for line in Path(name).read_text().splitlines() if line.strip()]
        meta = next((l for l in lines if l.get("meta")), {"servo": {}, "label": "?"})
        print(f"\n== {name} ({meta['label']})")
        summarize([l for l in lines if not l.get("meta")], meta["servo"])


def r_squared(xs: list[float], ys: list[float]) -> float:
    """How much of ys a straight line in xs explains."""
    if len(xs) < 3 or len(set(xs)) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy * sxy / (sxx * syy) if syy else float("nan")


AXES = (("tilt", "cy", "tracking_tilt_gain"), ("pan", "cx", "tracking_gain"))


def summarize(rows: list[dict], servo: dict) -> None:
    faces = [r for r in rows if r.get("faces") == 1]
    print(f"passes: {len(rows)}, with exactly one face: {len(faces)}, "
          f"with none: {sum(r.get('faces') == 0 for r in rows)}")
    ages = [r["age_ms"] for r in rows if r.get("age_ms", -1) >= 0]
    if ages:
        print(f"frame age when read: median {statistics.median(ages):.0f} ms")
    timed = [r for r in rows if "frame_age_ms" in r]
    if timed:
        med = lambda key: statistics.median(r[key] for r in timed)
        times = [r["t"] for r in rows]
        rate = (len(rows) - 1) / (times[-1] - times[0]) if len(rows) > 1 and times[-1] > times[0] else 0
        print(f"per pass (median): frame {med('frame_age_ms'):.0f} ms old when copied, "
              f"waited {med('wait_ms'):.0f} ms for it, "
              f"detection {med('process_ms'):.0f} ms; {rate:.2f} passes/s")
    if len(faces) < 3:
        print("not enough single-face passes to say anything")
        return
    for axis, centre, gain_key in AXES:
        summarize_axis(rows, faces, axis, centre, servo.get(gain_key))


def summarize_axis(rows: list[dict], faces: list[dict], axis: str, centre: str, gain) -> None:
    angles = [r[axis] for r in faces]
    lo, hi = min(angles), max(angles)
    limit = axis == "tilt" and (lo <= TILT_MIN + 0.5 or hi >= TILT_MAX - 0.5)
    print(f"\n-- {axis}: {lo:.1f}..{hi:.1f} deg"
          + (f"  (at a limit: {TILT_MIN:g}/{TILT_MAX:g})" if limit else ""))

    pairs = [(a, b) for a, b in zip(rows, rows[1:])
             if a.get("faces") == 1 and b.get("faces") == 1 and b["seq"] == a["seq"] + 1]
    steps, still = [], []
    for a, b in pairs:     # skipping gaps and dropouts: the tracker may have done anything
        ea, eb = a[centre] - 0.5, b[centre] - 0.5
        moved = b[axis] - a[axis]
        size = (a["h"] + b["h"]) / 2
        if abs(ea) > DEADBAND and abs(moved) >= STILL_DEG:
            steps.append((size, eb / ea, moved, ea))
        elif abs(moved) < STILL_DEG:
            still.append((size, abs(b[centre] - a[centre])))
    # Moving while the error sat inside the deadband: the tracker shouldn't.
    idle_moves = sum(abs(a[centre] - 0.5) <= DEADBAND and abs(b[axis] - a[axis]) >= STILL_DEG
                     for a, b in pairs)

    print(f"{'face size':>12} {'passes':>7} {'corrections':>12} {'step ratio':>11} "
          f"{'crossed':>8} {'still jitter':>13}")
    for lo_b, hi_b in zip(SIZE_BUCKETS, SIZE_BUCKETS[1:]):
        n = sum(lo_b <= r["h"] < hi_b for r in faces)
        if not n:
            continue
        s = [x for x in steps if lo_b <= x[0] < hi_b]
        j = [x[1] for x in still if lo_b <= x[0] < hi_b]
        ratio = f"{statistics.median(x[1] for x in s):+.2f}" if s else "-"
        crossed = f"{sum(x[1] < 0 for x in s)}/{len(s)}" if s else "-"
        jitter = f"{statistics.median(j):.3f} (max {max(j):.3f})" if j else "-"
        print(f"{lo_b:>5.2f}-{min(hi_b, 1.0):<4.2f}  {n:>7} {len(s):>12} {ratio:>11} {crossed:>8} {jitter:>13}")
    if idle_moves:
        print(f"moved {idle_moves} times with the error inside the deadband: the tracker hunting")

    if len(pairs) >= 5:
        errors = [b[centre] - 0.5 for _, b in pairs]
        same = r_squared([b[axis] for _, b in pairs], errors)
        prev = r_squared([a[axis] for a, _ in pairs], errors)
        verdict = "matches its own pass" if same >= prev else "LAGS a pass: the pose is stale"
        print(f"image vs pose: R^2 same pass {same:.2f}, previous pass {prev:.2f} -> {verdict}")
    if steps:
        moved_per_error = statistics.median(x[2] / x[3] for x in steps)
        print(f"{axis} moved {moved_per_error:+.1f} deg per unit of error "
              f"(configured gain {gain}; the sign is the servo's direction)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.101.40")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--label", default="run")
    p.set_defaults(fn=record)
    p = sub.add_parser("analyze")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=lambda a: load(a.files))
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
