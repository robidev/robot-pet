"""
Who the pet knows, and who is in front of it right now (PLAN.md 4.7, E3, E6).

- Faces are recognized on the PC (memory/recognition.py), which calls
  recognized() when a visit's vote names someone. Identity is sticky per
  presence episode: once named, the person counts as present until the
  debounced FacesPresence says nobody is there any more.
- A face is known by its fingerprints in the database (face_embeddings), as
  many per person as useful. The head's own recognizer and its 7 face slots
  are retired: people enrolled there enroll again.
- Enrollment takes ~5 good crops close up, asks the person to step back, and
  takes ~5 more: a face's fingerprint drifts with its size (E6a).
- Arrivals turn into `[event]` turns for the brain (a greeting at most once
  per `greet_every_h`), and so does a stranger: a face seen clearly several
  times that the recognizer couldn't name.
- With recognition off (no models, or recognition.enabled false) every face
  is unknown, so nobody counts as a stranger and no new faces can be learned.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import TYPE_CHECKING, Callable, Optional

from ..brain.tools import ToolError
from ..events import FacesChanged, FacesPresence, PersonArrived, PersonLeft
from ..vision.matching import class_centre, classify, normalized, to_blob
from .db import MemoryDB, Person

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

# Tier 0 is anyone with a stored face I haven't got to know yet. Calling it
# "new" made the pet tell someone it had recognized that they were new.
FAMILIARITY_WORDS = ("not familiar yet", "acquaintance", "regular", "favourite test subject")
SIGHTING_EVERY_S = 30.0
# Enrollment's second set counts as a step back when faces are at most this
# share of the close-up height (a real step from ~0.6 m is well under it).
STEP_BACK = 0.8


class People:
    def __init__(self, pet: "App", db: MemoryDB):
        self.pet = pet
        self.db = db
        self.cfg = pet.cfg.memory
        # person id -> Person, for everyone recognized in this presence episode.
        self.present: dict[int, Person] = {}
        self.faces_in_view = 0
        # Queues an [event] turn for the brain; see set_notify().
        self._notify: Optional[Callable[[str], None]] = None
        self._pending: list[str] = []
        self._last_sighting: dict[Optional[int], float] = {}
        self._last_stranger_note = float("-inf")
        self._enrolling = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        sub = self.pet.bus.subscribe(FacesChanged, FacesPresence)
        self._task = asyncio.create_task(self._watch(sub), name="people")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    # --- who is here ------------------------------------------------------------

    def who_is_here(self) -> tuple[list[str], int]:
        """(names of known people in view, how many other faces)."""
        names = [p.name for p in self.present.values()]
        return names, max(0, self.faces_in_view - len(names))

    def sole_person(self) -> Optional[Person]:
        """The one known person in view, if nobody else is; speech is probably theirs."""
        names, strangers = self.who_is_here()
        if len(names) == 1 and strangers == 0:
            return next(iter(self.present.values()))
        return None

    async def _watch(self, sub) -> None:
        async for event in sub:
            try:
                if isinstance(event, FacesChanged):
                    self._on_faces(event)
                elif not event.present:
                    self._on_nobody()
            except Exception:  # noqa: BLE001 - one bad event must not stop the watcher
                log.exception("people watcher failed on %r", event)

    def _on_faces(self, event: FacesChanged) -> None:
        self.faces_in_view = len(event.faces)
        if event.faces and not self.present:
            self._sighting(None, event.t, max(f.confidence for f in event.faces))

    def recognized(self, person_id: int, t: float, similarity: float) -> None:
        """The recognizer named a face in view."""
        person = self.db.person(person_id)
        if person is None:
            return                      # forgotten meanwhile
        self._sighting(person.id, t, similarity)
        if person.id not in self.present:
            self._arrived(person, t)

    def _on_nobody(self) -> None:
        self.faces_in_view = 0
        now = time.time()
        for person in self.present.values():
            self.db.mark_seen(person.id, now)
            self.pet.bus.publish(PersonLeft(person_id=person.id, name=person.name))
        self.present.clear()

    def _arrived(self, person: Person, now: float) -> None:
        away_s = None if person.last_seen_at is None else now - person.last_seen_at
        x, y = self._where()
        self.db.mark_seen(person.id, now, x, y)
        # A return after a proper absence is an interaction; flickering in and
        # out of view while sitting on the couch is not.
        if away_s is None or away_s > 3600:
            person = self.db.count_interaction(person.id, self._thresholds())
        self.present[person.id] = person
        log.info("recognized %s", person.name)
        self.pet.bus.publish(PersonArrived(person_id=person.id, name=person.name))

        greeted_ago = None if person.last_greeted_at is None else now - person.last_greeted_at
        if greeted_ago is None or greeted_ago > self.cfg.greet_every_h * 3600:
            self.db.mark_greeted(person.id, now)
            self._tell(f"{person.name} just came into view. {describe(person, now, away_s)}")

    def stranger_seen(self) -> None:
        """The recognizer saw a face clearly, several times, and couldn't name it."""
        now = time.time()
        if not self.pet.recognition_on:
            return
        if self.present or not self.pet.face or not self.pet.face.presence.present:
            return
        if now - self._last_stranger_note < self.cfg.stranger_every_min * 60:
            return
        self._last_stranger_note = now
        self._tell("Someone you don't recognize has been in front of you for a few seconds. "
                   "If you want to know who they are, ask; you can offer to remember their face.")

    def _sighting(self, person_id: Optional[int], t: float, face_conf: float) -> None:
        if t - self._last_sighting.get(person_id, float("-inf")) < SIGHTING_EVERY_S:
            return
        self._last_sighting[person_id] = t
        x, y = self._where()
        # The robot's own position, until C5 projects the face onto the map.
        self.db.add_sighting(person_id, t, x, y, pose_conf=None, face_conf=face_conf)

    def _where(self) -> tuple[Optional[float], Optional[float]]:
        """The robot's position in the reference map's frame (spatial/frame.py), if known."""
        vacuum, frame = self.pet.vacuum, self.pet.frame
        pose = vacuum.state.pose if vacuum is not None else None
        spot = frame.to_reference_now(pose.x, pose.y) if pose and frame else None
        return spot if spot else (None, None)

    def _thresholds(self) -> list[tuple[int, int]]:
        return [tuple(t) for t in self.cfg.familiarity_thresholds]

    def set_notify(self, notify: Callable[[str], None]) -> None:
        """Connects the brain; anything that happened before it was up is passed on now."""
        self._notify = notify
        pending, self._pending = self._pending, []
        for text in pending:
            notify(text)

    def _tell(self, text: str) -> None:
        if self._notify is not None:
            self._notify(text)
        else:
            self._pending.append(text)

    # --- faces -------------------------------------------------------------------

    async def enroll(self, name: str, insist: bool = False) -> str:
        name = " ".join(name.split())
        if not name:
            raise ToolError("I need a name to file the face under")
        recognizer = self.pet.recognizer
        if recognizer is None:
            raise ToolError("face recognition is switched off, so I can't learn or recognize "
                            "faces at the moment")
        face = self.pet.face
        if face is None or not face.state.reachable:
            raise ToolError("my head is offline, so I can't see anyone to remember")
        if self._enrolling.locked():
            raise ToolError("I'm already memorizing a face; one at a time")
        async with self._enrolling:
            recognizer.paused = True
            try:
                return await self._enroll(recognizer, name, insist)
            finally:
                recognizer.paused = False

    async def _enroll(self, recognizer, name: str, insist: bool) -> str:
        rc = self.pet.cfg.recognition
        existing = self.db.person_by_name(name)
        if existing and self.db.face_count(existing.id) and not insist:
            raise ToolError(f"I already have {existing.name}'s face stored. If I keep failing to "
                            "recognize them, call again with insist=true to take a fresh one.")

        close, trouble = await recognizer.collect(rc.enroll_samples, rc.enroll_timeout_s)
        if trouble["crowd"] >= 2:
            raise ToolError("I can see more than one face. Only the person I should remember "
                            "can be in front of me, or I might store the wrong one.")
        if len(close) < 3:
            if trouble["small"] > trouble["none"]:
                raise ToolError("I can see a face, but too small or too dark to learn. Ask them "
                                "to come a bit closer, facing me, where there's some light.")
            raise ToolError("I can't see a face right now. They need to stand in front of my "
                            "camera, facing me, fairly close.")

        # Someone I know already? Their face goes to them, not to a new name.
        others = {pid: c for pid, c in recognizer.centres.items()
                  if existing is None or pid != existing.id}
        mean = normalized(class_centre([s.embedding for s in close]))
        guess = classify(mean, others, rc.unknown_sim, rc.margin)
        if guess.person_id is not None and guess.similarity >= rc.accept_sim and not insist:
            match = self.db.person(guess.person_id)
            raise ToolError(f"this face looks like {match.name} to me. If it really is someone "
                            "else, call again with insist=true.")

        # A second set a step back: a face's fingerprint drifts with its size.
        # Only crops that really are smaller count: the first live enrollment
        # took both sets at ~220 px, which only added near-copies.
        far: list = []
        close_px = statistics.median(s.height for s in close)
        if close_px >= 2 * rc.min_face_px and self.pet.speaker:
            for prompt in ("Now take one step back, and keep looking at me.",
                           "A bit further back, please. One more step."):
                self.pet.speaker.say(prompt)
                await asyncio.sleep(rc.enroll_step_back_s)
                got, _ = await recognizer.collect(rc.enroll_samples, rc.enroll_timeout_s)
                far = [s for s in got if s.height <= STEP_BACK * close_px]
                if len(far) >= 3:
                    break
                log.info("enrollment: no step back yet (%s px against %.0f close up)",
                         [round(s.height) for s in got], close_px)
            else:
                far = []

        person = existing or self.db.add_person(name)
        if existing:
            self.db.delete_face_embeddings(person.id)       # a retake replaces the old face
        for s in close + far:
            self.db.add_face_embedding(person.id, to_blob(s.embedding), source="enroll",
                                       face_px=s.height, sharpness=s.sharpness,
                                       brightness=s.brightness)
        recognizer.reload()
        recognizer.mark_current(person.id)

        now = time.time()
        self.db.mark_seen(person.id, now, *self._where())
        # They're in front of me and we're mid-conversation: no greeting now.
        self.db.mark_greeted(person.id, now)
        self.present[person.id] = self.db.person(person.id)
        log.info("enrolled %s: %d close, %d a step back", person.name, len(close), len(far))
        stored = (f"Done: I've stored {person.name}'s face ({len(close) + len(far)} views) and will "
                  "recognize them from now on.")
        if self.pet.speaker and close_px >= 2 * rc.min_face_px and not far:
            stored += (" Only close up, though: they didn't step back. From further away it may "
                       "take me a moment longer, until I've seen them there.")
        return stored

    async def forget(self, name: str) -> str:
        person = self.db.person_by_name(name)
        if person is None:
            raise ToolError(f"I don't know anyone called {name}")
        self.db.delete_person(person.id)            # fingerprints go with them
        if self.pet.recognizer is not None:
            self.pet.recognizer.reload()
        self.present.pop(person.id, None)
        log.warning("forgot %s", person.name)
        return f"Forgotten: {person.name}'s face and everything I knew about them."


def describe(person: Person, now: float, away_s: Optional[float] = None) -> str:
    """One line about someone, for the brain."""
    bits = []
    if person.familiarity > 0:
        bits.append(f"Familiarity: {FAMILIARITY_WORDS[min(person.familiarity, 3)]}.")
    if person.nickname:
        bits.append(f"You call them {person.nickname}.")
    if away_s is not None:
        bits.append(f"Last seen {ago(away_s)} ago.")
    elif person.last_seen_at is None:
        bits.append("First time you've seen them since learning their face.")
    return " ".join(bits)


def ago(seconds: float) -> str:
    if seconds < 90:
        return "a moment"
    if seconds < 3600:
        return f"{round(seconds / 60)} minutes"
    if seconds < 36 * 3600:
        hours = round(seconds / 3600)
        return "an hour" if hours == 1 else f"{hours} hours"
    days = round(seconds / 86400)
    return "a day" if days == 1 else f"{days} days"
