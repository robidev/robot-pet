"""
Who the pet knows, and who is in front of it right now (PLAN.md 4.7, E3).

- Identity is sticky per presence episode. Recognition flickers between an
  id and -1 from frame to frame (similarity sits close to the threshold),
  so once a face is recognized the person counts as present until the
  debounced FacesPresence says nobody is there any more.
- The database owns the face slot -> person mapping. The device's ids are
  neither contiguous nor reused in order after a delete, so the mapping is
  reconciled against /api/face/list rather than assumed.
- Enrollment diffs /api/face/list before and after arming, instead of
  trusting whichever id shows up next in a face event.
- Arrivals turn into `[event]` turns for the brain (a greeting at most once
  per `greet_every_h`), and so does a stranger who stays in view.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional

from ..brain.tools import ToolError
from ..events import FacesChanged, FacesPresence, PersonArrived, PersonLeft
from .db import MemoryDB, Person

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

# Tier 0 is anyone with a stored face I haven't got to know yet. Calling it
# "new" made the pet tell someone it had recognized that they were new.
FAMILIARITY_WORDS = ("not familiar yet", "acquaintance", "regular", "favourite test subject")
SIGHTING_EVERY_S = 30.0


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
        self._stranger_timer: Optional[asyncio.TimerHandle] = None
        self._last_stranger_note = float("-inf")
        self._enrolling = asyncio.Lock()
        # Pace of enrollment's frame sampling and slot polling (tests set 0).
        self.poll_s = 0.2
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        sub = self.pet.bus.subscribe(FacesChanged, FacesPresence)
        self._task = asyncio.create_task(self._watch(sub), name="people")
        try:
            await self.reconcile()
        except Exception as exc:  # noqa: BLE001 - the head may still be booting
            log.warning("could not reconcile face slots yet: %s", exc)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
        if self._stranger_timer:
            self._stranger_timer.cancel()

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
        now = event.t
        for face in event.faces:
            if not face.recognized:
                continue
            person = self.db.person_by_slot(face.id)
            if person is None:
                continue
            self._sighting(person.id, now, face.confidence)
            if person.id not in self.present:
                self._arrived(person, now)
        if event.faces and not self.present:
            self._sighting(None, now, max(f.confidence for f in event.faces))
            if self._stranger_timer is None:
                loop = asyncio.get_running_loop()
                self._stranger_timer = loop.call_later(self.cfg.stranger_after_s, self._stranger)

    def _on_nobody(self) -> None:
        self.faces_in_view = 0
        if self._stranger_timer:
            self._stranger_timer.cancel()
            self._stranger_timer = None
        now = time.time()
        for person in self.present.values():
            self.db.mark_seen(person.id, now)
            self.pet.bus.publish(PersonLeft(person_id=person.id, name=person.name))
        self.present.clear()

    def _arrived(self, person: Person, now: float) -> None:
        if self._stranger_timer:
            self._stranger_timer.cancel()
            self._stranger_timer = None
        away_s = None if person.last_seen_at is None else now - person.last_seen_at
        x, y = self._where()
        self.db.mark_seen(person.id, now, x, y)
        # A return after a proper absence is an interaction; flickering in and
        # out of view while sitting on the couch is not.
        if away_s is None or away_s > 3600:
            person = self.db.count_interaction(person.id, self._thresholds())
        self.present[person.id] = person
        log.info("recognized %s (slot %s)", person.name, person.face_slot)
        self.pet.bus.publish(PersonArrived(person_id=person.id, name=person.name))

        greeted_ago = None if person.last_greeted_at is None else now - person.last_greeted_at
        if greeted_ago is None or greeted_ago > self.cfg.greet_every_h * 3600:
            self.db.mark_greeted(person.id, now)
            self._tell(f"{person.name} just came into view. {describe(person, now, away_s)}")

    def _stranger(self) -> None:
        self._stranger_timer = None
        now = time.time()
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
        vacuum = self.pet.vacuum
        pose = vacuum.state.pose if vacuum is not None else None
        return (pose.x, pose.y) if pose else (None, None)

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

    # --- face slots -------------------------------------------------------------

    async def reconcile(self) -> list[int]:
        """
        Drops DB slots the device no longer has (the person keeps their name
        and notes, just not their face) and returns device ids no one is
        attached to.
        """
        face = self.pet.face
        if face is None:
            return []
        on_device = set(await face.list_enrolled())
        for person in self.db.people():
            if person.face_slot is not None and person.face_slot not in on_device:
                log.warning("face slot %d of %s is gone from the device; unlinking",
                            person.face_slot, person.name)
                self.db.set_face_slot(person.id, None)
        known = {p.face_slot for p in self.db.people() if p.face_slot is not None}
        orphans = sorted(on_device - known)
        if orphans:
            log.info("enrolled on the device but unnamed: %s", orphans)
        return orphans

    async def enroll(self, name: str, insist: bool = False) -> str:
        name = " ".join(name.split())
        if not name:
            raise ToolError("I need a name to file the face under")
        face = self.pet.face
        if face is None or not face.state.reachable:
            raise ToolError("my head is offline, so I can't see anyone to remember")
        if self._enrolling.locked():
            raise ToolError("I'm already memorizing a face; one at a time")
        async with self._enrolling:
            return await self._enroll(face, name, insist)

    async def _enroll(self, face, name: str, insist: bool) -> str:
        existing = self.db.person_by_name(name)
        if existing and existing.face_slot is not None and not insist:
            raise ToolError(f"I already have {existing.name}'s face stored. If I keep failing to "
                            "recognize them, call again with insist=true to take a fresh one.")

        counts, ids = await self._look_closely(face)
        if max(counts, default=0) == 0:
            raise ToolError("I can't see a face right now. They need to stand in front of my "
                            "camera, facing me, fairly close.")
        if max(counts) > 1:
            raise ToolError("I can see more than one face. Only the person I should remember "
                            "can be in front of me, or I might store the wrong one.")

        seen_as = ids.most_common(1)[0][0] if ids else None
        if seen_as is not None and ids[seen_as] >= 2:
            match = self.db.person_by_slot(seen_as)
            if match is None:
                # Enrolled before the DB knew about it (or its row was lost).
                return self._name_slot(name, existing, seen_as, adopted=True)
            if existing and match.id == existing.id and not insist:
                raise ToolError(f"that is {match.name}, and I already know their face")
            if match.id != (existing.id if existing else None) and not insist:
                raise ToolError(f"this face looks like {match.name} to me. If it really is "
                                f"someone else, call again with insist=true.")

        before = set(await face.list_enrolled())
        if len(before) >= self.cfg.face_slots:
            stored = [p.name for p in self.db.people() if p.face_slot in before]
            raise ToolError(f"my face memory is full ({len(before)} faces: "
                            f"{', '.join(stored) or 'none of them named'}). I'd have to forget "
                            "someone first, and only if asked to.")

        await face.enroll_next_face()
        new_ids: set = set()
        deadline = time.monotonic() + self.cfg.enroll_timeout_s
        try:
            while time.monotonic() < deadline and not new_ids:
                await asyncio.sleep(self.poll_s)
                new_ids = set(await face.list_enrolled()) - before
        finally:
            if not new_ids:
                await face.cancel_enroll()
        if not new_ids:
            raise ToolError("I didn't manage to get a good look. Ask them to face me, "
                            "hold still and come a little closer, then try again.")
        slot = min(new_ids)
        if existing and existing.face_slot is not None:
            # A retake: the old enrollment was evidently not working.
            try:
                await face.delete_enrolled(existing.face_slot)
            except Exception as exc:  # noqa: BLE001 - it's being replaced either way
                log.warning("could not delete old slot %s: %s", existing.face_slot, exc)
        return self._name_slot(name, existing, slot, adopted=False)

    async def _look_closely(self, face, samples: int = 5) -> tuple[list[int], Counter]:
        """A second of frames: how many faces each had, and which ids they were seen as."""
        counts: list[int] = []
        ids: Counter = Counter()
        for index in range(samples):
            frame = await face.current_faces()
            counts.append(len(frame.faces))
            ids.update(f.id for f in frame.faces if f.recognized)
            if index < samples - 1:
                await asyncio.sleep(self.poll_s)
        return counts, ids

    def _name_slot(self, name: str, existing: Optional[Person], slot: int, adopted: bool) -> str:
        if existing is not None:
            self.db.set_face_slot(existing.id, slot)
            person = self.db.person(existing.id)
        else:
            person = self.db.add_person(name, slot)
        now = time.time()
        self.db.mark_seen(person.id, now, *self._where())
        # They're in front of me and we're mid-conversation: no greeting now.
        self.db.mark_greeted(person.id, now)
        self.present[person.id] = self.db.person(person.id)
        log.info("face slot %d is now %s%s", slot, person.name, " (adopted)" if adopted else "")
        if adopted:
            return (f"I already had this face stored, just without a name. It's {person.name} "
                    "now, and I'll recognize them from here on.")
        return f"Done: I've stored {person.name}'s face and will recognize them from now on."

    async def forget(self, name: str) -> str:
        person = self.db.person_by_name(name)
        if person is None:
            raise ToolError(f"I don't know anyone called {name}")
        if person.face_slot is not None and self.pet.face is not None:
            try:
                await self.pet.face.delete_enrolled(person.face_slot)
            except Exception as exc:  # noqa: BLE001 - the slot may already be gone
                log.warning("deleting slot %s for %s: %s", person.face_slot, person.name, exc)
        self.db.delete_person(person.id)
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
