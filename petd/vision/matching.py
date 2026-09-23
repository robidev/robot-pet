"""
Who a fingerprint belongs to, and who a visit's attempts add up to (PLAN.md
4.7, E6). Numbers only, no models: the rules come from Frigate's face
recognition, the thresholds from E6a (config.recognition).

- Each person's centre is a trimmed mean of their fingerprints, after dropping
  any far from the rest (Frigate's build_class_mean), so one bad or mislabeled
  sample doesn't drag it off.
- One attempt names its best person only if it's at least `unknown_sim`
  similar and `margin` ahead of the next one; otherwise it's unknown.
- A visit's name is a vote over its attempts, each weighted by face area and
  by how far it is above `unknown_sim`: a close, clear look outweighs several
  distant ones. It needs `min_agree` agreeing attempts, no tie, and a weighted
  mean of at least `accept_sim`. Otherwise nobody: greeting a guest by someone
  else's name is the mistake that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

AREA_CAP = 100.0 * 100.0           # a face 100 px tall counts fully; bigger doesn't count more


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, np.float32)


def normalized(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def class_centre(vectors: list[np.ndarray], trim: float = 0.15, outlier_sim: float = 0.30,
                 min_keep: float = 0.7) -> np.ndarray:
    """A person's centre, robust to a few bad samples. With fewer than 5, just trimmed."""
    arr = np.stack(vectors)
    if len(arr) >= 5:
        keep = np.ones(len(arr), bool)
        floor = max(5, int(np.ceil(min_keep * len(arr))))
        for _ in range(3):
            sims = arr @ normalized(trimmed_mean(arr[keep], trim))
            new_keep = sims >= outlier_sim
            if new_keep.sum() < floor:
                new_keep = np.zeros(len(arr), bool)
                new_keep[np.argsort(-sims)[:floor]] = True
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        arr = arr[keep]
    return normalized(trimmed_mean(arr, trim))


def trimmed_mean(arr: np.ndarray, trim: float) -> np.ndarray:
    """Per dimension, the mean without the lowest and highest `trim` share."""
    cut = int(len(arr) * trim)
    if len(arr) - 2 * cut < 1:
        return arr.mean(0)
    return np.sort(arr, axis=0)[cut:len(arr) - cut].mean(0)


@dataclass(frozen=True)
class Guess:
    person_id: Optional[int]        # None = unknown
    similarity: float               # to the best person, named or not
    best_id: Optional[int]          # who the best person was, even if not named
    second: float                   # similarity to the next person (-1 if none)


def classify(embedding: np.ndarray, centres: dict[int, np.ndarray],
             unknown_sim: float, margin: float) -> Guess:
    if not centres:
        return Guess(None, 0.0, None, -1.0)
    sims = sorted(((float(embedding @ c), pid) for pid, c in centres.items()), reverse=True)
    best_sim, best_id = sims[0]
    second = sims[1][0] if len(sims) > 1 else -1.0
    named = best_sim >= unknown_sim and best_sim - second >= margin
    return Guess(best_id if named else None, best_sim, best_id, second)


@dataclass
class Vote:
    """One face's attempts over a visit."""
    attempts: list = field(default_factory=list)      # (Guess, face area)

    def add(self, guess: Guess, area: float) -> None:
        self.attempts.append((guess, area))

    def decide(self, unknown_sim: float, accept_sim: float, min_agree: int) -> Optional[tuple[int, float]]:
        """(person id, weighted mean similarity), or None: not sure enough."""
        counts: dict[int, int] = {}
        weighted: dict[int, float] = {}
        weights: dict[int, float] = {}
        for guess, area in self.attempts:
            if guess.person_id is None:
                continue
            weight = min(area, AREA_CAP) * (guess.similarity - unknown_sim) * 10
            pid = guess.person_id
            counts[pid] = counts.get(pid, 0) + 1
            weighted[pid] = weighted.get(pid, 0.0) + guess.similarity * weight
            weights[pid] = weights.get(pid, 0.0) + weight
        if not counts:
            return None
        best = max(weighted, key=weighted.get)
        if counts[best] < min_agree:
            return None
        if any(pid != best and n == counts[best] for pid, n in counts.items()):
            return None                 # as many votes for someone else: not sure
        mean = weighted[best] / weights[best] if weights[best] > 0 else 0.0
        return (best, mean) if mean >= accept_sim else None
