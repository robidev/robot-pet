"""
What the pet has stored: people, places, facts, conversations, photos.

    .venv/bin/python scripts/show_memory.py                  # everything, briefly
    .venv/bin/python scripts/show_memory.py conversation 12  # one conversation, word for word
    .venv/bin/python scripts/show_memory.py conversation     # the latest one
    .venv/bin/python scripts/show_memory.py map              # places on the robot's map, as a PNG

Read-only: the database (memory.db_path, runtime/pet.db by default) is
opened read-only, so it's safe while petd runs. `map` fetches the live
map from Valetudo and marks the named places, and where the robot was when
it last saw each person (not where they stood: petd doesn't place faces on
the map yet); the dock is green, the robot blue.

Faces are stored as fingerprints (128 numbers each), not images; the count
per person is shown, by source (enroll, or grown from confident recognitions).
Photos show up under "photos" once petd keeps some (the observations table).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vacuum-api"))

from petd.config import load_config  # noqa: E402
from petd.memory.people import FAMILIARITY_WORDS, ago  # noqa: E402


def when(t) -> str:
    if t is None:
        return "never"
    return f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t))} ({ago(time.time() - t)} ago)"


def where(x, y) -> str:
    return "?" if x is None else f"({x:.0f}, {y:.0f})"


def overview(db: sqlite3.Connection) -> None:
    people = db.execute("SELECT * FROM people ORDER BY name").fetchall()
    face_counts: dict = {}
    has_faces = db.execute("SELECT 1 FROM sqlite_master WHERE name = 'face_embeddings'").fetchone()
    if has_faces:           # created by petd on its next start after E6 (read-only here)
        for row in db.execute("SELECT person_id, source, count(*) AS n FROM face_embeddings "
                              "GROUP BY person_id, source"):
            face_counts.setdefault(row["person_id"], {})[row["source"]] = row["n"]
    print(f"== People ({len(people)})")
    for p in people:
        sightings = db.execute("SELECT count(*) FROM sightings WHERE person_id = ?", (p["id"],)).fetchone()[0]
        bits = [f"familiarity: {FAMILIARITY_WORDS[min(p['familiarity'], 3)]}",
                f"{p['interactions']} interactions", f"{sightings} sightings"]
        faces = face_counts.get(p["id"], {})
        face = (f"face stored: {sum(faces.values())} fingerprints "
                f"({', '.join(f'{n} {src}' for src, n in sorted(faces.items()))})" if faces
                else "no face stored")
        print(f"- {p['name']}" + (f' ("{p["nickname"]}")' if p["nickname"] else "") + f": {face}")
        print(f"    {', '.join(bits)}")
        print(f"    last seen {when(p['last_seen_at'])}, from {where(p['last_seen_x'], p['last_seen_y'])}, "
              f"last greeted {when(p['last_greeted_at'])}")
        if p["notes"]:
            print(f"    notes: {p['notes']}")
        for f in db.execute("SELECT text FROM facts WHERE about = ? ORDER BY t_utc", (p["id"],)):
            print(f"    fact: {f['text']}")
    strangers = db.execute("SELECT count(*), max(t_utc) FROM sightings WHERE person_id IS NULL").fetchone()
    if strangers[0]:
        print(f"- (unrecognized faces: {strangers[0]} sightings, last {when(strangers[1])})")

    places = db.execute("SELECT * FROM places ORDER BY name").fetchall()
    print(f"\n== Places ({len(places)}), in map cm")
    for p in places:
        print(f"- {p['name']}: {where(p['x'], p['y'])}, named {when(p['created_at'])}")

    facts = db.execute("SELECT * FROM facts WHERE about IS NULL ORDER BY t_utc").fetchall()
    print(f"\n== Facts ({len(facts)})")
    for f in facts:
        print(f"- {f['text']}  ({when(f['t_utc'])})")

    conversations = db.execute(
        "SELECT c.*, count(u.id) AS n FROM conversations c "
        "LEFT JOIN utterances u ON u.conversation_id = c.id GROUP BY c.id ORDER BY c.id").fetchall()
    print(f"\n== Conversations ({len(conversations)}); `conversation N` shows one")
    for c in conversations:
        length = "ongoing" if c["ended_at"] is None else f"{(c['ended_at'] - c['started_at']) / 60:.0f} min"
        print(f"- #{c['id']} {time.strftime('%Y-%m-%d %H:%M', time.localtime(c['started_at']))}, "
              f"{length}, {c['n']} utterances" + (f": {c['summary']}" if c["summary"] else ""))

    photos = db.execute("SELECT * FROM observations ORDER BY t_utc").fetchall()
    print(f"\n== Photos ({len(photos)})")
    for o in photos:
        path = Path(o["jpeg_path"]) if o["jpeg_path"] else None
        if path is not None and not path.is_absolute():
            path = ROOT / path
        state = "missing" if path is not None and not path.exists() else ""
        print(f"- {when(o['t_utc'])} at {where(o['x'], o['y'])}: {o['caption'] or ''}\n    {path or '(no file)'} {state}")

    kv = db.execute("SELECT * FROM kv ORDER BY key").fetchall()
    if kv:
        print(f"\n== Other ({len(kv)})")
        for row in kv:
            print(f"- {row['key']}: {row['value']}")


def conversation(db: sqlite3.Connection, number) -> None:
    if number is None:
        row = db.execute("SELECT max(id) FROM conversations").fetchone()
        number = row[0]
    c = db.execute("SELECT * FROM conversations WHERE id = ?", (number,)).fetchone()
    if c is None:
        sys.exit(f"no conversation #{number}")
    print(f"#{c['id']}, {when(c['started_at'])}")
    if c["summary"]:
        print(f"journal: {c['summary']}")
    print()
    for u in db.execute("SELECT * FROM utterances WHERE conversation_id = ? ORDER BY t_utc", (number,)):
        print(f"{time.strftime('%H:%M:%S', time.localtime(u['t_utc']))}  {u['speaker']:>8}: {u['text']}")


def draw_map(db: sqlite3.Connection, cfg, out: Path) -> None:
    from valetudo_client import ValetudoClient
    markers = [(p["x"], p["y"], p["name"]) for p in db.execute("SELECT * FROM places")]
    markers += [(p["last_seen_x"], p["last_seen_y"], f"{p['name']} seen from here")
                for p in db.execute("SELECT * FROM people WHERE last_seen_x IS NOT NULL")]
    image = ValetudoClient(cfg.vacuum.host, cfg.vacuum.port).get_map_image(markers=markers)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"{len(markers)} markers -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config (default: ./config.yaml if present)")
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("conversation")
    p.add_argument("number", type=int, nargs="?")
    p = sub.add_parser("map")
    p.add_argument("--out", type=Path, default=ROOT / "runtime" / "memory-map.png")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    path = cfg.path(cfg.memory.db_path)
    if not path.exists():
        sys.exit(f"no database at {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    if args.cmd == "conversation":
        conversation(db, args.number)
    elif args.cmd == "map":
        draw_map(db, cfg, args.out)
    else:
        overview(db)


if __name__ == "__main__":
    main()
