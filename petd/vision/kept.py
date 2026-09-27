"""
The faces behind the numbers (PLAN.md 4.7, E6c, first part): the aligned
112x112 crops recognition worked on, as small JPEGs on the PC.

    runtime/faces/attempts/<time>_as-<verdict>_best-<name>-<similarity>.jpg
        every usable recognition attempt, the last `keep_attempts`
    runtime/faces/fingerprints/<id>.jpg
        every stored fingerprint (enrolled or grown), by its face_embeddings id;
        a crop goes when its fingerprint does (prune_fingerprints)

So a similarity in petd.log, or a fingerprint in the database, can be looked
at (scripts/show_memory.py faces draws them as contact sheets), and a misread
corrected (scripts/fix_faces.py). Photos of people: runtime/ stays out of git.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

FINGERPRINT_NAME = re.compile(r"^(\d+)\.jpg$")
# <time>_as-<verdict>_best-<name>-<similarity>[-n].jpg; names may hold "-", and
# a similarity may be negative ("best-Robin--0.01").
ATTEMPT_NAME = re.compile(r"^(?P<time>\d{8}-\d{6}\.\d{3})_as-(?P<verdict>.+?)_best-(?P<best>.+?)"
                          r"-(?P<similarity>-?\d+\.\d+)(?:-\d+)?\.jpg$")


@dataclass
class Attempt:
    """A kept attempt, read back from its file name."""
    path: Path
    time: str               # YYYYmmdd-HHMMSS.mmm, local
    verdict: Optional[str]  # the name it was taken for, None if unknown
    best: str               # the closest person, named or not
    similarity: float

    @classmethod
    def parse(cls, path: Path) -> Optional["Attempt"]:
        match = ATTEMPT_NAME.match(path.name)
        if not match:
            return None
        verdict = match["verdict"]
        return cls(path, match["time"], None if verdict == "unknown" else verdict, match["best"],
                   float(match["similarity"]))

    @property
    def clock(self) -> str:
        """HH:MM:SS"""
        return f"{self.time[9:11]}:{self.time[11:13]}:{self.time[13:15]}"


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
        name = f"{stamp}_as-{verdict}_best-{self.name_of(guess.best_id)}-{guess.similarity:.2f}"
        path, n = self.attempts / f"{name}.jpg", 1
        while path.exists():                            # two faces in one snapshot
            path, n = self.attempts / f"{name}-{n}.jpg", n + 1
        self._save(sample.crop, path)
        self._prune()

    def fingerprint(self, embedding_id: int, sample) -> None:
        if sample.crop is not None:
            self._save(sample.crop, self.fingerprints / f"{embedding_id}.jpg")

    def recent_attempts(self, count: Optional[int] = None) -> list[Attempt]:
        """The kept attempts, oldest first; the newest `count` if given."""
        found = [a for a in (Attempt.parse(p) for p in sorted(self.attempts.glob("*.jpg"))) if a]
        return found[-count:] if count else found

    def find_attempt(self, name: str) -> Optional[Attempt]:
        """An attempt by file name, or by the start of it (its time is enough)."""
        exact = self.attempts / Path(name).name
        if exact.is_file():
            return Attempt.parse(exact)
        matches = [a for a in self.recent_attempts() if a.path.name.startswith(Path(name).name)]
        return matches[0] if len(matches) == 1 else None

    def fingerprint_crop(self, embedding_id: int) -> Optional[Path]:
        path = self.fingerprints / f"{embedding_id}.jpg"
        return path if path.is_file() else None

    def prune_fingerprints(self, kept_ids: set) -> int:
        """Removes the crops of fingerprints no longer stored (replaced, re-enrolled, forgotten)."""
        removed = 0
        for path in self.fingerprints.iterdir():
            match = FINGERPRINT_NAME.match(path.name)
            if match and path.is_file() and int(match.group(1)) not in kept_ids:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def _save(self, crop, path: Path) -> None:
        try:
            crop.save(path, quality=90)
        except OSError as exc:
            log.warning("couldn't keep a face crop at %s: %s", path, exc)

    def _prune(self) -> None:
        kept = sorted(self.attempts.glob("*.jpg"))      # names start with the time
        for old in kept[:max(0, len(kept) - self.keep_attempts)]:
            old.unlink(missing_ok=True)


def contact_sheet(tiles: list[tuple[Optional[Path], str]], out: Path, columns: int = 10,
                  heading: Optional[str] = None) -> Path:
    """
    The crops side by side, each with its label underneath (a crop that is
    missing gets a grey square); rows of `columns`. A tile of None, "" ends
    the row early, which keeps each person's fingerprints on rows of their own.
    """
    from PIL import Image, ImageDraw
    size, label_h, top = 112, 22, 24 if heading else 0
    rows: list[list] = [[]]
    for tile in tiles:
        if tile == (None, ""):
            if rows[-1]:
                rows.append([])
            continue
        if len(rows[-1]) == columns:
            rows.append([])
        rows[-1].append(tile)
    rows = [row for row in rows if row] or [[]]
    sheet = Image.new("RGB", (columns * size, top + len(rows) * (size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    if heading:
        draw.text((4, 6), heading, fill="black")
    for r, row in enumerate(rows):
        for c, (path, label) in enumerate(row):
            x, y = c * size, top + r * (size + label_h)
            if path is not None and path.is_file():
                with Image.open(path) as crop:
                    sheet.paste(crop.convert("RGB").resize((size, size)), (x, y))
            else:
                draw.rectangle((x, y, x + size - 1, y + size - 1), fill=(200, 200, 200))
            draw.text((x + 3, y + size + 5), label[:19], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out
