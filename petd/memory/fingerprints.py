"""
Correcting stored faces by hand (PLAN.md 4.7, E6c): forget a wrong
fingerprint, or give a misread attempt to the right person. Used by
scripts/fix_faces.py; the kept crops (vision/kept.py) show what to correct.

A running petd picks the changes up at its next reload of the stored faces
(an enrollment, a grown look, a forget) or its next start.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..vision.matching import class_centre, from_blob, to_blob
from .recognition import Recognizer

if TYPE_CHECKING:
    from ..config import RecognitionConfig
    from ..vision.kept import Attempt, FaceKeeper
    from .db import MemoryDB, Person

SAME_LOOK = 0.95    # a stored fingerprint this alike to an attempt was grown from that same look


class Refused(Exception):
    """A correction that would do harm, unless forced."""


@dataclass
class Assigned:
    embedding_id: int
    similarity: float                       # to the person's stored face, before adding
    replaced: Optional[int] = None          # a grown fingerprint traded out (the set was full)
    elsewhere: list = field(default_factory=list)   # (person name, fingerprint id): the same look


def forget_fingerprints(db: "MemoryDB", ids: list[int], keeper: Optional["FaceKeeper"] = None,
                        force: bool = False) -> list[str]:
    """Deletes these fingerprints (and their crops). Refuses to leave someone with none, unless forced."""
    rows = {}
    for embedding_id in ids:
        row = db.face_embedding(embedding_id)
        if row is None:
            raise Refused(f"no fingerprint {embedding_id}")
        rows[embedding_id] = row
    by_person: dict[int, int] = {}
    for row in rows.values():
        by_person[row["person_id"]] = by_person.get(row["person_id"], 0) + 1
    for person_id, count in by_person.items():
        if count >= db.face_count(person_id) and not force:
            name = db.person(person_id).name
            raise Refused(f"that would leave {name} with no stored face (enroll again instead, "
                          "or --force)")
    done = []
    for embedding_id, row in rows.items():
        db.delete_face_embedding(embedding_id)
        done.append(f"forgot fingerprint {embedding_id} ({row['source']}) of "
                    f"{db.person(row['person_id']).name}")
    if keeper is not None:
        keeper.prune_fingerprints(db.face_embedding_ids())
    return done


def assign_attempt(db: "MemoryDB", engine, cfg: "RecognitionConfig", attempt: "Attempt",
                   person: "Person", keeper: Optional["FaceKeeper"] = None,
                   force: bool = False) -> Assigned:
    """
    Adds a kept attempt's crop to `person` as an 'assigned' fingerprint (kept
    like an enrolled one: growth never trades it out). Refuses a crop unlike
    their stored face (below unknown_sim) or one they already have, unless forced.
    """
    from PIL import Image
    with Image.open(attempt.path) as crop:
        vector = engine.embed(crop.convert("RGB").resize((112, 112)))
    rows = db.face_rows(person.id)
    vectors = np.stack([from_blob(blob) for _, _, blob in rows]) if rows else np.zeros((0, 128))
    similarity = float(class_centre(list(vectors)) @ vector) if rows else 1.0
    if rows and not force:
        if similarity < cfg.unknown_sim:
            raise Refused(f"this crop is {similarity:.2f} like {person.name}'s stored face, under "
                          f"unknown_sim ({cfg.unknown_sim}): is it really {person.name}? (--force)")
        closest = float((vectors @ vector).max())
        if closest >= cfg.duplicate_sim:
            raise Refused(f"{person.name} already has this look ({closest:.2f} to a stored "
                          "fingerprint): nothing to add (--force)")

    replaced = None
    if len(rows) >= cfg.max_per_person:
        victim = Recognizer._most_redundant(rows, vectors)
        if victim is not None:
            db.delete_face_embedding(victim[0])
            replaced = victim[0]
    embedding_id = db.add_face_embedding(person.id, to_blob(vector), source="assigned")
    if keeper is not None:
        shutil.copyfile(attempt.path, keeper.fingerprints / f"{embedding_id}.jpg")
        keeper.prune_fingerprints(db.face_embedding_ids())

    elsewhere = []
    for other_id, person_id, blob in db.all_face_rows():
        if person_id != person.id and float(from_blob(blob) @ vector) >= SAME_LOOK:
            elsewhere.append((db.person(person_id).name, other_id))
    return Assigned(embedding_id, similarity, replaced, elsewhere)
