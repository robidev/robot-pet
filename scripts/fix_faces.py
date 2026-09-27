"""
Correct what the pet stored about faces (PLAN.md 4.7, E6c). Find what to fix
with `show_memory.py faces`, which lists every fingerprint by id and draws the
recent recognition attempts.

    .venv/bin/python scripts/fix_faces.py forget 41 42        # wrong fingerprints out of a set
    .venv/bin/python scripts/fix_faces.py assign 20260927-101722 Noah
                                                              # a misread attempt to the right person

`forget` won't leave anyone without a stored face, and `assign` won't add a
crop unlike the person's stored face or a look they already have, unless
--force. `assign` also names any fingerprint grown from that same look under
someone else, the usual companion of a misread: forget that one too.

Writes to the database (memory.db_path). A running petd picks the changes up
at its next reload of the stored faces (an enrollment, a grown look, a
forget) or its next start.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from petd.config import load_config  # noqa: E402
from petd.memory.db import MemoryDB  # noqa: E402
from petd.memory.fingerprints import Refused, assign_attempt, forget_fingerprints  # noqa: E402
from petd.vision.kept import FaceKeeper  # noqa: E402


def petd_running(cfg) -> bool:
    try:
        with socket.create_connection((cfg.api.host, cfg.api.port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config (default: ./config.yaml if present)")
    parser.add_argument("--force", action="store_true", help="do it even when refused")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("forget", help="delete fingerprints by id")
    p.add_argument("ids", type=int, nargs="+")
    p = sub.add_parser("assign", help="add a kept attempt to a person")
    p.add_argument("attempt", help="its file name, or the start of it (the time is enough)")
    p.add_argument("name")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    path = cfg.path(cfg.memory.db_path)
    if not path.exists():
        sys.exit(f"no database at {path}")
    db = MemoryDB(path)
    rc = cfg.recognition
    keeper = FaceKeeper(cfg.path(rc.faces_dir), rc.keep_attempts, str)
    try:
        if args.cmd == "forget":
            for line in forget_fingerprints(db, args.ids, keeper, force=args.force):
                print(line)
        else:
            attempt = keeper.find_attempt(args.attempt)
            if attempt is None:
                sys.exit(f"no single kept attempt matches {args.attempt!r} (see show_memory.py faces)")
            person = db.person_by_name(args.name)
            if person is None:
                sys.exit(f"nobody called {args.name!r}; to add someone new, enroll them")
            from petd.vision.faces import FaceEngine
            engine = FaceEngine(cfg.path(rc.models_dir), rc.detector_model, rc.recognizer_model)
            done = assign_attempt(db, engine, rc, attempt, person, keeper, force=args.force)
            print(f"{attempt.path.name} -> {person.name}: fingerprint {done.embedding_id} "
                  f"({done.similarity:.2f} to {person.name}'s stored face)")
            if done.replaced is not None:
                print(f"  {person.name}'s set was full: grown fingerprint {done.replaced} made room")
            for name, other in done.elsewhere:
                print(f"  the same look is also stored under {name} as fingerprint {other}: "
                      f"`fix_faces.py forget {other}` if that was the misread")
    except Refused as exc:
        sys.exit(f"refused: {exc}")
    finally:
        db.close()
    if petd_running(cfg):
        print("petd is running: it picks this up at its next reload of the stored faces, or restart it")


if __name__ == "__main__":
    main()
