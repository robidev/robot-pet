"""
The faces behind the numbers (PLAN.md 4.7, E6c, first part): the aligned
112x112 crops recognition worked on, as small JPEGs on the PC.

    runtime/faces/attempts/<time>_as-<verdict>_best-<name>-<similarity>.jpg
        every usable recognition attempt, the last `keep_attempts`
    runtime/faces/fingerprints/<id>.jpg
        every stored fingerprint (enrolled or grown), by its face_embeddings id

So a similarity in petd.log, or a fingerprint in the database, can be looked
at. Photos of people: runtime/ stays out of git.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


class FaceKeeper:
    def __init__(self, root: Path, keep_attempts: int, name_of: Callable[[Optional[int]], str]):
        self.attempts = Path(root) / "attempts"
        self.fingerprints = Path(root) / "fingerprints"
        self.keep_attempts = keep_attempts
        self.name_of = name_of
        self.attempts.mkdir(parents=True, exist_ok=True)
        self.fingerprints.mkdir(parents=True, exist_ok=True)

    def attempt(self, sample, guess) -> None:
        if sample.crop is None:
            return
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
        verdict = self.name_of(guess.person_id) if guess.person_id is not None else "unknown"
        name = f"{stamp}_as-{verdict}_best-{self.name_of(guess.best_id)}-{guess.similarity:.2f}.jpg"
        self._save(sample.crop, self.attempts / name)
        self._prune()

    def fingerprint(self, embedding_id: int, sample) -> None:
        if sample.crop is not None:
            self._save(sample.crop, self.fingerprints / f"{embedding_id}.jpg")

    def _save(self, crop, path: Path) -> None:
        try:
            crop.save(path, quality=90)
        except OSError as exc:
            log.warning("couldn't keep a face crop at %s: %s", path, exc)

    def _prune(self) -> None:
        kept = sorted(self.attempts.glob("*.jpg"))      # names start with the time
        for old in kept[:max(0, len(kept) - self.keep_attempts)]:
            old.unlink(missing_ok=True)
