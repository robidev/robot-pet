"""
Who is in front of the pet (PLAN.md 4.7, E6): face recognition on the PC,
once per visit rather than per frame.

When a face appears and not everyone in view is known yet, a visit starts:
one head snapshot every `attempt_every_s`, each face in it fingerprinted
(petd/vision/faces.py) and compared with everyone's centre. A face is followed
from snapshot to snapshot by where it is, and named by a vote over its attempts
(petd/vision/matching.py) as soon as the vote is sure; People then greets as
before. Up to `max_attempts` a visit, and `confirm_attempts` more once everyone
is named, which is also when confident attempts grow the person's fingerprint
set: daylight, lamp light and new angles come with ordinary use.

Names stick for the presence episode, but a fresh look replaces them
(People.still_here) when a face goes out of view (the head seeing fewer faces
than before, even for a moment: someone may have stepped out and someone else
in within the presence debounce), and every `recheck_every_s` while a named
face is in view, so an early wrong name gets corrected. A fresh look that
can't name a face drops its old name too: unsure means unknown.

A face seen clearly several times without a name is a stranger (People says so,
at most every `stranger_every_min`). A face too small, dark or doubtful never
gets a name at all: "unknown" is the safe side.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..events import FacesChanged, FacesPresence
from ..vision.matching import Vote, class_centre, classify, from_blob, to_blob

if TYPE_CHECKING:
    from ..app import App
    from ..vision.faces import FaceSample
    from ..vision.kept import FaceKeeper

log = logging.getLogger(__name__)

TRACK_DISTANCE = 0.2        # normalized: further than this from a known face is another face
STRANGER_ATTEMPTS = 3       # usable attempts without a name before someone is a stranger
RESTART_AFTER_S = 30.0      # after a visit gave up, the same faces aren't tried again for this long


@dataclass
class Track:
    """One face over a visit."""
    centre: tuple[float, float]
    first_seen: float
    vote: Vote = field(default_factory=Vote)
    samples: list = field(default_factory=list)      # (FaceSample, Guess, considered for growth?)
    person_id: Optional[int] = None
    grown: int = 0                                   # fingerprints kept from this visit


class Recognizer:
    def __init__(self, pet: "App", engine, keeper: Optional["FaceKeeper"] = None):
        self.pet = pet
        self.engine = engine
        self.keeper = keeper            # the crops behind attempts and fingerprints, if kept
        self.cfg = pet.cfg.recognition
        self.centres: dict[int, np.ndarray] = {}
        self.tracks: list[Track] = []
        self.paused = False             # enrollment takes the snapshots meanwhile
        self.collect_every_s = 0.3      # between enrollment snapshots (tests shorten it)
        self._visit: Optional[asyncio.Task] = None
        self._task: Optional[asyncio.Task] = None
        self._periodic: Optional[asyncio.Task] = None
        self._gave_up_at = float("-inf")
        self._gave_up_faces = 0
        self._faces_before = 0
        self._recheck = False           # a face went out of view: look again before trusting names

    async def start(self) -> None:
        self.reload()
        log.info("face recognition on: %d people with a stored face", len(self.centres))
        sub = self.pet.bus.subscribe(FacesChanged, FacesPresence)
        self._task = asyncio.create_task(self._watch(sub), name="recognizer")
        self._periodic = asyncio.create_task(self._recheck_now_and_then(), name="recognition-recheck")

    async def close(self) -> None:
        for task in (self._task, self._periodic, self._visit):
            if task:
                task.cancel()

    def reload(self) -> None:
        """Everyone's centre, from the database (after enrolling, growing, forgetting)."""
        db = self.pet.db
        stored = db.face_embeddings() if db is not None else {}
        self.centres = {pid: class_centre([from_blob(b) for b in blobs]) for pid, blobs in stored.items()}
        if self.keeper is not None and db is not None:
            self.keeper.prune_fingerprints(db.face_embedding_ids())

    def knows_face(self, person_id: int) -> bool:
        return person_id in self.centres

    # --- visits -----------------------------------------------------------------

    async def _watch(self, sub) -> None:
        async for event in sub:
            try:
                if isinstance(event, FacesPresence):
                    if not event.present:
                        self._end_visit()
                    continue
                faces = len(event.faces)
                if faces < self._faces_before and self.tracks:
                    self._recheck = True
                self._faces_before = faces
                if faces:
                    self._maybe_start(faces)
            except Exception:  # noqa: BLE001 - one bad event must not stop recognition
                log.exception("recognizer failed on %r", event)

    async def _recheck_now_and_then(self) -> None:
        while True:
            every = self.cfg.recheck_every_s
            await asyncio.sleep(every if every > 0 else 1.0)
            if (every > 0 and self._faces_before and not self._visiting()
                    and any(t.person_id is not None for t in self.tracks)):
                self._recheck = True
                self._maybe_start(self._faces_before)

    def _visiting(self) -> bool:
        return self._visit is not None and not self._visit.done()

    def _maybe_start(self, faces: int) -> None:
        if self._visiting() or self.paused:
            return
        if not self._recheck and faces <= sum(t.person_id is not None for t in self.tracks):
            return                      # everyone in view is known already
        if (time.monotonic() - self._gave_up_at < RESTART_AFTER_S
                and faces <= self._gave_up_faces):
            return                      # just tried these faces; a new one would restart
        recheck, self._recheck = self._recheck, False
        if recheck:
            log.debug("recognition: looking again (a face went out of view, or it's time)")
            self.tracks = []
        self._visit = asyncio.create_task(self._run_visit(recheck), name="recognition-visit")

    def _end_visit(self) -> None:
        if self._visit:
            self._visit.cancel()
        self.tracks = []
        self._gave_up_at = float("-inf")
        self._recheck = False

    def _after_visit(self) -> None:
        """A face went out of view during the visit just ended: look again now."""
        if self._recheck and self._faces_before:
            self._maybe_start(self._faces_before)

    async def _run_visit(self, recheck: bool = False) -> None:
        attempts = confirmed = 0
        try:
            while attempts < self.cfg.max_attempts + self.cfg.confirm_attempts:
                if not self.paused:
                    await self.attempt()
                    attempts += 1
                    everyone = bool(self.tracks) and all(t.person_id is not None for t in self.tracks)
                    if everyone:
                        confirmed += 1
                        if recheck:
                            self._settle()      # quick: the first visit already grew the set
                            return
                        if confirmed >= self.cfg.confirm_attempts:
                            return
                    elif attempts >= self.cfg.max_attempts:
                        break
                await asyncio.sleep(self.cfg.attempt_every_s)
            self._gave_up_at = time.monotonic()
            self._gave_up_faces = len(self.tracks)
            log.info("recognition: gave up on %d of %d faces after %d attempts",
                     sum(t.person_id is None for t in self.tracks), len(self.tracks), attempts)
            if recheck:
                self._settle()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the next visit will try again
            log.exception("recognition visit failed")
        finally:
            asyncio.get_running_loop().call_soon(self._after_visit)

    def _settle(self) -> None:
        """After a fresh look: whoever it didn't name is no longer taken to be in view."""
        if self.pet.people is not None:
            self.pet.people.still_here({t.person_id for t in self.tracks if t.person_id is not None})

    async def attempt(self) -> None:
        """One snapshot: every usable face in it fingerprinted, voted on, maybe named."""
        samples = await self.snapshot_faces()
        now = time.time()
        for sample in samples:
            if not self.usable(sample):
                continue
            guess = classify(sample.embedding, self.centres, self.cfg.unknown_sim, self.cfg.margin)
            track = self._track_for(sample, now)
            track.vote.add(guess, sample.area)
            track.samples.append([sample, guess, False])
            log.debug("face at (%.2f, %.2f), %.0f px: best %s at %.2f (next %.2f) -> %s",
                      *sample.centre, sample.height, guess.best_id, guess.similarity, guess.second,
                      guess.person_id)
            if self.keeper is not None:
                self.keeper.attempt(sample, guess)
        for track in self.tracks:
            decision = track.vote.decide(self.cfg.unknown_sim, self.cfg.accept_sim, self.cfg.min_agree)
            if decision is not None and decision[0] != track.person_id:
                track.person_id, similarity = decision
                log.info("recognized person %d (similarity %.2f over %d attempts)",
                         track.person_id, similarity, len(track.vote.attempts))
                if self.pet.people is not None:
                    self.pet.people.recognized(track.person_id, now, similarity)
            if track.person_id is not None:
                self._grow(track)
            elif (len(track.samples) >= STRANGER_ATTEMPTS and self.pet.people is not None
                  and now - track.first_seen >= self.pet.cfg.memory.stranger_after_s):
                self.pet.people.stranger_seen()

    async def snapshot_faces(self) -> list:
        face = self.pet.face
        if face is None:
            return []
        try:
            snap = await face.snapshot()
        except Exception as exc:  # noqa: BLE001 - the head may be busy or away
            log.debug("recognition: no snapshot (%s)", exc)
            return []
        if snap is None or not snap.jpeg:
            return []
        return await asyncio.to_thread(self.engine.analyse, snap.jpeg)

    def usable(self, sample: "FaceSample") -> bool:
        """Skip rather than guess: too small, too doubtful a face, too dark."""
        return (sample.height >= self.cfg.min_face_px and sample.score >= self.cfg.min_detection_score
                and sample.brightness >= self.cfg.min_brightness)

    def _track_for(self, sample: "FaceSample", now: float) -> Track:
        cx, cy = sample.centre
        nearest = min(self.tracks, default=None,
                      key=lambda t: (t.centre[0] - cx) ** 2 + (t.centre[1] - cy) ** 2)
        if nearest is not None and ((nearest.centre[0] - cx) ** 2
                                    + (nearest.centre[1] - cy) ** 2) ** 0.5 < TRACK_DISTANCE:
            nearest.centre = (cx, cy)
            return nearest
        track = Track(centre=(cx, cy), first_seen=now)
        self.tracks.append(track)
        return track

    def _grow(self, track: Track) -> None:
        """
        Keep a confident attempt as another fingerprint, if it's a new look:
        less than duplicate_sim to every one kept, at most grow_per_visit a
        visit. When the person has max_per_person already, a newer look takes
        the place of their most redundant grown fingerprint (the one closest to
        another); enrolled ones stay.
        """
        db, pid, cfg = self.pet.db, track.person_id, self.cfg
        if db is None:
            return
        changed = False
        for entry in track.samples:
            sample, guess, considered = entry
            if considered or guess.person_id != pid or guess.similarity < cfg.grow_sim:
                continue
            entry[2] = True
            if track.grown >= cfg.grow_per_visit:
                break
            rows = db.face_rows(pid)
            vectors = np.stack([from_blob(blob) for _, _, blob in rows]) if rows else np.zeros((0, 128))
            closest = float((vectors @ sample.embedding).max()) if rows else -1.0
            if closest >= cfg.duplicate_sim:
                continue                    # a look already kept
            if len(rows) >= cfg.max_per_person:
                victim = self._most_redundant(rows, vectors)
                if victim is None or victim[1] <= closest:
                    continue                # nothing kept is more redundant than this look would be
                db.delete_face_embedding(victim[0])
            embedding_id = db.add_face_embedding(pid, to_blob(sample.embedding), source="grown",
                                                 face_px=sample.height, sharpness=sample.sharpness,
                                                 brightness=sample.brightness)
            if self.keeper is not None:
                self.keeper.fingerprint(embedding_id, sample)
            track.grown += 1
            changed = True
            log.info("person %d: kept a new look (%.0f px, %.2f to the closest kept)",
                     pid, sample.height, closest)
        if changed:
            self.reload()

    @staticmethod
    def _most_redundant(rows: list, vectors: np.ndarray) -> Optional[tuple[int, float]]:
        """(id, similarity to its closest other) of the grown fingerprint most like another one."""
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -1.0)
        grown = [i for i, (_, source, _) in enumerate(rows) if source == "grown"]
        if not grown:
            return None
        worst = max(grown, key=lambda i: sims[i].max())
        return rows[worst][0], float(sims[worst].max())

    def mark_current(self, person_id: int) -> None:
        """The one face in view is this person now (just enrolled): no greeting, no stranger."""
        if len(self.tracks) == 1:
            self.tracks[0].person_id = person_id
        elif not self.tracks:
            self.tracks.append(Track(centre=(0.5, 0.5), first_seen=time.time(), person_id=person_id))

    # --- enrollment ---------------------------------------------------------------

    async def collect(self, count: int, timeout_s: float) -> tuple[list, dict]:
        """
        Up to `count` usable faces of the one person in view, from separate
        snapshots. Also returns what got in the way: {'crowd', 'small', 'none'}
        counts, for telling the person what to do.
        """
        got: list = []
        trouble = {"crowd": 0, "small": 0, "none": 0}
        deadline = time.monotonic() + timeout_s
        while len(got) < count and time.monotonic() < deadline:
            faces = [s for s in await self.snapshot_faces()
                     if s.score >= self.cfg.min_detection_score]
            if len(faces) > 1:
                trouble["crowd"] += 1
                if trouble["crowd"] >= 2:
                    break
            elif not faces:
                trouble["none"] += 1
            elif not self.usable(faces[0]):
                trouble["small"] += 1
            else:
                got.append(faces[0])
            await asyncio.sleep(self.collect_every_s)
        return got, trouble


class FakeEngine:
    """For --fake and tests: analyse() returns scripted faces, whatever the snapshot."""

    def __init__(self):
        self.frames: list[list] = []        # consumed one per snapshot
        self.default: list = []             # once they run out

    def analyse(self, jpeg: bytes) -> list:
        return self.frames.pop(0) if self.frames else list(self.default)


def sample(embedding, centre: tuple[float, float] = (0.5, 0.4), height: float = 120.0,
           score: float = 0.93, brightness: float = 150.0):
    """A FaceSample for tests and --fake, with a given fingerprint."""
    from ..vision.faces import FaceSample
    width, image = height * 0.8, (640, 480)
    x, y = centre[0] * image[0] - width / 2, centre[1] * image[1] - height / 2
    vector = np.asarray(embedding, np.float32)
    return FaceSample(box=(x, y, width, height), score=score, embedding=vector / np.linalg.norm(vector),
                      brightness=brightness, sharpness=200.0, image_size=image)
