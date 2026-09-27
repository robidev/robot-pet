"""
The pet's long-term memory: one SQLite file (runtime/pet.db, PLAN.md 4.7).

Synchronous on purpose. Every query here touches a handful of rows in a
local file and takes well under a millisecond, so calling it straight from
the event loop is cheaper than a thread hop. Anything that could grow
(utterances, sightings) is only ever appended or read with a LIMIT.

Schema changes go in MIGRATIONS as a new entry; PRAGMA user_version
records how many have been applied.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MIGRATIONS = [
    """
    CREATE TABLE people (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        face_slot INTEGER UNIQUE,            -- enrolled id on the face device
        created_at REAL NOT NULL,
        last_seen_at REAL,
        last_seen_x REAL,
        last_seen_y REAL,
        last_greeted_at REAL,
        notes TEXT NOT NULL DEFAULT '',
        nickname TEXT,
        familiarity INTEGER NOT NULL DEFAULT 0,
        interactions INTEGER NOT NULL DEFAULT 0,
        face_z_m REAL
    );
    CREATE TABLE places (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        x REAL NOT NULL, y REAL NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE TABLE sightings (
        id INTEGER PRIMARY KEY,
        person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
        t_utc REAL NOT NULL,
        x REAL, y REAL,
        pose_conf REAL, face_conf REAL,
        source TEXT NOT NULL DEFAULT 'face'
    );
    CREATE INDEX sightings_person_t ON sightings(person_id, t_utc);
    CREATE TABLE conversations (
        id INTEGER PRIMARY KEY,
        started_at REAL NOT NULL,
        ended_at REAL,
        summary TEXT
    );
    CREATE TABLE utterances (
        id INTEGER PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
        t_utc REAL NOT NULL,
        speaker TEXT NOT NULL,               -- 'pet' | a name | 'someone' | 'event'
        text TEXT NOT NULL
    );
    CREATE INDEX utterances_conversation ON utterances(conversation_id);
    CREATE TABLE facts (
        id INTEGER PRIMARY KEY,
        t_utc REAL NOT NULL,
        about INTEGER REFERENCES people(id) ON DELETE CASCADE,  -- NULL = general
        text TEXT NOT NULL
    );
    CREATE INDEX facts_about ON facts(about, t_utc);
    CREATE TABLE observations (
        id INTEGER PRIMARY KEY,
        t_utc REAL NOT NULL,
        room TEXT, x REAL, y REAL,
        caption TEXT,
        jpeg_path TEXT
    );
    CREATE TABLE kv (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """,
    # Face recognition moved to the PC (PLAN.md 4.7, E6): fingerprints here,
    # as many per person as useful. people.face_slot (the head's own
    # recognizer) is no longer used.
    """
    CREATE TABLE face_embeddings (
        id INTEGER PRIMARY KEY,
        person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
        t_utc REAL NOT NULL,
        vector BLOB NOT NULL,                -- SFace: 128 float32, normalized
        face_px REAL,                        -- face height in the snapshot
        sharpness REAL,
        brightness REAL,
        source TEXT NOT NULL                 -- 'enroll' | 'grown' | 'assigned' (by hand)
    );
    CREATE INDEX face_embeddings_person ON face_embeddings(person_id);
    """,
]

MAX_NOTES_CHARS = 1000
MAX_FACT_CHARS = 300


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    face_slot: Optional[int]
    created_at: float
    last_seen_at: Optional[float]
    last_seen_x: Optional[float]
    last_seen_y: Optional[float]
    last_greeted_at: Optional[float]
    notes: str
    nickname: Optional[str]
    familiarity: int
    interactions: int
    face_z_m: Optional[float]


@dataclass(frozen=True)
class Fact:
    id: int
    t_utc: float
    about: Optional[int]
    text: str


@dataclass(frozen=True)
class Conversation:
    id: int
    started_at: float
    ended_at: Optional[float]
    summary: Optional[str]


class MemoryDB:
    def __init__(self, path: Path | str):
        """path=':memory:' gives a throwaway database (tests, --fake)."""
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self._db.close()

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        for index in range(version, len(MIGRATIONS)):
            self._db.executescript("BEGIN;" + MIGRATIONS[index]
                                   + f"; PRAGMA user_version = {index + 1}; COMMIT;")

    # --- people ---------------------------------------------------------------

    def add_person(self, name: str, face_slot: Optional[int] = None) -> Person:
        cur = self._db.execute(
            "INSERT INTO people (name, face_slot, created_at) VALUES (?, ?, ?)",
            (name.strip(), face_slot, time.time()))
        return self.person(cur.lastrowid)

    def person(self, person_id: int) -> Optional[Person]:
        row = self._db.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
        return Person(**row) if row else None

    def person_by_name(self, name: str) -> Optional[Person]:
        """Case-insensitive; also matches a nickname."""
        name = name.strip()
        row = self._db.execute(
            "SELECT * FROM people WHERE name = ? COLLATE NOCASE "
            "OR nickname = ? COLLATE NOCASE ORDER BY name = ? COLLATE NOCASE DESC LIMIT 1",
            (name, name, name)).fetchone()
        return Person(**row) if row else None

    def person_by_slot(self, face_slot: int) -> Optional[Person]:
        row = self._db.execute("SELECT * FROM people WHERE face_slot = ?", (face_slot,)).fetchone()
        return Person(**row) if row else None

    def people(self) -> list[Person]:
        rows = self._db.execute(
            "SELECT * FROM people ORDER BY familiarity DESC, last_seen_at DESC").fetchall()
        return [Person(**row) for row in rows]

    def set_face_slot(self, person_id: int, face_slot: Optional[int]) -> None:
        self._db.execute("UPDATE people SET face_slot = ? WHERE id = ?", (face_slot, person_id))

    def delete_person(self, person_id: int) -> None:
        self._db.execute("DELETE FROM people WHERE id = ?", (person_id,))

    # --- face fingerprints (petd/vision) ------------------------------------------

    def add_face_embedding(self, person_id: int, vector: bytes, *, source: str,
                           face_px: Optional[float] = None, sharpness: Optional[float] = None,
                           brightness: Optional[float] = None, t: Optional[float] = None) -> int:
        return self._db.execute(
            "INSERT INTO face_embeddings (person_id, t_utc, vector, face_px, sharpness, "
            "brightness, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (person_id, time.time() if t is None else t, vector, face_px, sharpness,
             brightness, source)).lastrowid

    def face_embedding(self, embedding_id: int) -> Optional[dict]:
        """One fingerprint's row (without the vector), or None."""
        row = self._db.execute("SELECT id, person_id, t_utc, face_px, source FROM face_embeddings "
                               "WHERE id = ?", (embedding_id,)).fetchone()
        return dict(row) if row else None

    def all_face_rows(self) -> list[tuple[int, int, bytes]]:
        """Everyone's fingerprints: (id, person id, vector)."""
        return [(row["id"], row["person_id"], row["vector"])
                for row in self._db.execute("SELECT id, person_id, vector FROM face_embeddings ORDER BY id")]

    def face_embedding_ids(self) -> set[int]:
        return {row["id"] for row in self._db.execute("SELECT id FROM face_embeddings")}

    def face_embeddings(self) -> dict[int, list[bytes]]:
        """person id -> their fingerprints, raw."""
        out: dict[int, list[bytes]] = {}
        for row in self._db.execute("SELECT person_id, vector FROM face_embeddings ORDER BY id"):
            out.setdefault(row["person_id"], []).append(row["vector"])
        return out

    def face_rows(self, person_id: int) -> list[tuple[int, str, bytes]]:
        """One person's fingerprints: (id, source, vector)."""
        return [(row["id"], row["source"], row["vector"]) for row in self._db.execute(
            "SELECT id, source, vector FROM face_embeddings WHERE person_id = ? ORDER BY id",
            (person_id,))]

    def delete_face_embedding(self, embedding_id: int) -> None:
        self._db.execute("DELETE FROM face_embeddings WHERE id = ?", (embedding_id,))

    def face_count(self, person_id: int) -> int:
        return self._db.execute("SELECT count(*) FROM face_embeddings WHERE person_id = ?",
                                (person_id,)).fetchone()[0]

    def delete_face_embeddings(self, person_id: int) -> None:
        self._db.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))

    def add_note(self, person_id: int, note: str) -> None:
        """Notes are a running paragraph; the oldest text falls off past the cap."""
        person = self.person(person_id)
        if person is None:
            return
        combined = (person.notes + " " + note.strip()).strip() if person.notes else note.strip()
        if len(combined) > MAX_NOTES_CHARS:
            combined = "…" + combined[-(MAX_NOTES_CHARS - 1):]
        self._db.execute("UPDATE people SET notes = ? WHERE id = ?", (combined, person_id))

    def set_nickname(self, person_id: int, nickname: Optional[str]) -> None:
        self._db.execute("UPDATE people SET nickname = ? WHERE id = ?",
                         (nickname.strip() if nickname else None, person_id))

    def mark_seen(self, person_id: int, t: float, x: Optional[float] = None,
                  y: Optional[float] = None) -> None:
        self._db.execute(
            "UPDATE people SET last_seen_at = ?, "
            "last_seen_x = COALESCE(?, last_seen_x), last_seen_y = COALESCE(?, last_seen_y) "
            "WHERE id = ?", (t, x, y, person_id))

    def mark_greeted(self, person_id: int, t: float) -> None:
        self._db.execute("UPDATE people SET last_greeted_at = ? WHERE id = ?", (t, person_id))

    def count_interaction(self, person_id: int, thresholds: list[tuple[int, int]]) -> Person:
        """
        One more interaction; familiarity rises to the highest tier whose
        (min_interactions, min_days_seen) are both met. It never falls.
        """
        self._db.execute("UPDATE people SET interactions = interactions + 1 WHERE id = ?",
                         (person_id,))
        person = self.person(person_id)
        days = self.days_seen(person_id)
        tier = 0
        for index, (min_interactions, min_days) in enumerate(thresholds, start=1):
            if person.interactions >= min_interactions and days >= min_days:
                tier = index
        if tier > person.familiarity:
            self._db.execute("UPDATE people SET familiarity = ? WHERE id = ?", (tier, person_id))
            person = self.person(person_id)
        return person

    # --- sightings ------------------------------------------------------------

    def add_sighting(self, person_id: Optional[int], t: float, x: Optional[float] = None,
                     y: Optional[float] = None, pose_conf: Optional[float] = None,
                     face_conf: Optional[float] = None, source: str = "face") -> None:
        self._db.execute(
            "INSERT INTO sightings (person_id, t_utc, x, y, pose_conf, face_conf, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (person_id, t, x, y, pose_conf, face_conf, source))

    def days_seen(self, person_id: int) -> int:
        """Distinct local calendar days with a sighting."""
        return self._db.execute(
            "SELECT COUNT(DISTINCT date(t_utc, 'unixepoch', 'localtime')) "
            "FROM sightings WHERE person_id = ?", (person_id,)).fetchone()[0]

    # --- facts ----------------------------------------------------------------

    def add_fact(self, text: str, about: Optional[int] = None) -> Fact:
        text = text.strip()[:MAX_FACT_CHARS]
        cur = self._db.execute("INSERT INTO facts (t_utc, about, text) VALUES (?, ?, ?)",
                               (time.time(), about, text))
        return Fact(cur.lastrowid, time.time(), about, text)

    def facts(self, about: Optional[int] = None, limit: int = 20) -> list[Fact]:
        """Newest first. about=None gives general facts (not about a person)."""
        if about is None:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE about IS NULL ORDER BY t_utc DESC LIMIT ?", (limit,))
        else:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE about = ? ORDER BY t_utc DESC LIMIT ?", (about, limit))
        return [Fact(**row) for row in rows.fetchall()]

    # --- conversations --------------------------------------------------------

    def start_conversation(self, t: Optional[float] = None) -> int:
        cur = self._db.execute("INSERT INTO conversations (started_at) VALUES (?)",
                               (t or time.time(),))
        return cur.lastrowid

    def end_conversation(self, conversation_id: int, summary: Optional[str] = None,
                         t: Optional[float] = None) -> None:
        self._db.execute("UPDATE conversations SET ended_at = ?, summary = ? WHERE id = ?",
                         (t or time.time(), summary, conversation_id))

    def add_utterance(self, conversation_id: int, speaker: str, text: str,
                      t: Optional[float] = None) -> None:
        self._db.execute(
            "INSERT INTO utterances (conversation_id, t_utc, speaker, text) VALUES (?, ?, ?, ?)",
            (conversation_id, t or time.time(), speaker, text))

    def utterances(self, conversation_id: int) -> list[tuple[str, str]]:
        rows = self._db.execute(
            "SELECT speaker, text FROM utterances WHERE conversation_id = ? ORDER BY id",
            (conversation_id,)).fetchall()
        return [(row["speaker"], row["text"]) for row in rows]

    def journal(self, limit: int = 5) -> list[Conversation]:
        """The most recent summarized conversations, oldest first."""
        rows = self._db.execute(
            "SELECT * FROM conversations WHERE summary IS NOT NULL AND summary != '' "
            "ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [Conversation(**row) for row in reversed(rows)]

    # --- places and kv ----------------------------------------------------------

    def set_place(self, name: str, x: float, y: float) -> None:
        self._db.execute(
            "INSERT INTO places (name, x, y, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET x = excluded.x, y = excluded.y",
            (name.strip(), x, y, time.time()))

    def place(self, name: str) -> Optional[tuple[float, float]]:
        row = self._db.execute("SELECT x, y FROM places WHERE name = ?", (name.strip(),)).fetchone()
        return (row["x"], row["y"]) if row else None

    def places(self) -> list[str]:
        return [row["name"] for row in self._db.execute("SELECT name FROM places ORDER BY name")]

    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: Optional[str]) -> None:
        self._db.execute("INSERT INTO kv (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
