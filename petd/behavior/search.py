"""
Looking around for someone (PLAN.md 4.6 search behaviour, E4, Step 5).

search() runs as the one motion (App.start_motion), so "stop" cancels it,
and it steers the head meanwhile (Expressions.claim).

The head pans slowly from one side to the other at face.search_tilt_deg, in
steps (the firmware jumps to a manual pose; a head standing still between
steps gives the detector sharper frames; jumpy is fine, Robin 2026-10-05).
On a face it goes back to the pose of the frame that showed it (the head
has moved on, ~1 s of detection), tracks it and gives recognition up to
search_naming_wait_s to name it:

- looking for anyone, the first face ends the search;
- looking for someone by name, a face that isn't them is passed over and the
  pan goes on; its direction is kept (relative to where the robot first
  faced) so it isn't checked again, also after a turn.

Off the dock, after a pass with nobody found, the body turns
search_turn_deg (closed loop, spatial/motion.py) and the head pans again,
search_body_turns times: with the head's ±90° that's the whole room. On the
dock it can't turn (the contacts), so the head pans to one end and back
search_sweeps times instead.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Optional

from ..events import FacesChanged
from ..spatial.person import bearing_left_deg
from .approach import head_claim

if TYPE_CHECKING:
    from ..app import App
    from ..memory.db import Person

log = logging.getLogger(__name__)

# The head moves in steps this far apart in time: 10 deg at 20 deg/s.
SEARCH_STEP_S = 0.5


def _named_here(pet: "App") -> list[str]:
    return pet.people.who_is_here()[0] if pet.people is not None else []


async def _wait_for_name(pet: "App", person: Optional["Person"], wait_s: float) -> bool:
    """Up to wait_s for recognition: the person named (with one), or anyone named (without)."""
    if pet.people is None or not pet.recognition_on:
        return person is None
    deadline = time.monotonic() + wait_s
    while True:
        if person is not None and person.id in pet.people.present:
            return True
        if person is None and _named_here(pet):
            return True
        if time.monotonic() >= deadline:
            return person is None       # anyone: a face is a find, named or not
        await asyncio.sleep(0.25)


class Search:
    def __init__(self, pet: "App", person: Optional["Person"]):
        self.pet = pet
        self.person = person
        self.cfg = pet.cfg.face
        self.cal = pet.cfg.calibration
        self.turned_deg = 0.0                   # the body, counter-clockwise, since the start
        self.checked: list[float] = []          # directions of faces that weren't them
        self.started = time.monotonic()
        self.ahead = False                      # nobody found: the head ends looking ahead

    @property
    def who(self) -> str:
        return self.person.name if self.person is not None else "anyone"

    def _known_face(self, face, pan: float) -> bool:
        direction = bearing_left_deg(face.cx, pan, self.cal) + self.turned_deg
        return any(abs(direction - seen) <= self.cfg.search_skip_deg for seen in self.checked)

    def _new_face(self, event: FacesChanged):
        return next((f for f in event.faces if not self._known_face(f, event.pan_deg)), None)

    async def run(self) -> tuple[str, bool]:
        pet, cfg = self.pet, self.cfg
        expressions = head_claim(pet)
        if expressions is not None:
            expressions.claim("search")
        sub = pet.bus.subscribe(FacesChanged)
        log.info("searching for %s", self.who)
        try:
            can_turn = pet.motion is not None and not pet.vacuum.state.docked
            turns = max(0, cfg.search_body_turns) if can_turn else 0
            for position in range(turns + 1):
                if position:
                    result = await pet.motion.turn_by(cfg.search_turn_deg)
                    self.turned_deg += result.done
                    if not result.ok and abs(result.done) < cfg.search_turn_deg / 2:
                        return self._not_found(f"I couldn't turn round: {result.reason}"), False
                found = await self._sweep(sub, passes=1 if can_turn else max(1, cfg.search_sweeps))
                if found:
                    return self._found(), True
            if can_turn:
                return self._not_found(f"I looked round with my head and turned {turns} times"), False
            why = ("I'm on my dock, so I can't turn round" if pet.vacuum.state.docked
                   else "I can't turn my body right now")
            return self._not_found(f"I only looked round with my head: {why}"), False
        finally:
            sub.close()
            if expressions is not None:
                expressions.release("search")
            try:
                if self.ahead:
                    await pet.face.set_servo(pan_deg=self.cal.pan_forward_deg, tilt_deg=self.cfg.search_tilt_deg)
                await pet.face.set_servo(mode="track")
            except Exception:  # noqa: BLE001 - the head may be gone
                log.debug("search: couldn't turn tracking on", exc_info=True)

    def _found(self) -> str:
        took = time.monotonic() - self.started
        if self.person is not None:
            return f"My search found {self.person.name} after {took:.0f} s; my head is following them now."
        names = _named_here(self.pet)
        whom = " and ".join(names) if names else "someone I don't recognise"
        return f"My search found a face after {took:.0f} s: {whom}. My head is following them now."

    def _not_found(self, how: str) -> str:
        self.ahead = True
        whom = self.person.name if self.person is not None else "anyone"
        passed = f" (I passed {len(self.checked)} other face{'s' if len(self.checked) != 1 else ''})" \
            if self.checked else ""
        return f"My search didn't find {whom}{passed}: {how}. I'm looking ahead again."

    async def _sweep(self, sub, passes: int) -> bool:
        """The head across the room `passes` times (to one end, then across); True on a find."""
        face, cfg = self.pet.face, self.cfg
        low, high = cfg.search_pan_min_deg, cfg.search_pan_max_deg
        tilt = cfg.search_tilt_deg
        pan = face.state.pan_deg if face.state.pan_deg is not None else self.cal.pan_forward_deg
        pan = max(low, min(high, pan))
        near, far = (low, high) if pan - low <= high - pan else (high, low)
        # Once: to the nearer end, then across. More: and back, `passes` times.
        legs = [near, far] if passes == 1 else [near] + [far, near] * passes
        await face.set_servo(mode="manual")
        await face.set_servo(pan_deg=pan, tilt_deg=tilt)
        for target in legs:
            while True:
                event = await self._pan_to(pan, target, sub)
                if event is None:
                    pan = target
                    break
                if await self._check(event, sub):
                    return True
                pan = event.pan_deg             # pan on from where that face was
                await face.set_servo(mode="manual")
                await face.set_servo(pan_deg=pan, tilt_deg=tilt)
        return False

    async def _check(self, event: FacesChanged, sub) -> bool:
        """A face: back to that frame's pose, tracking, and a moment for recognition."""
        pet, face = self.pet, self.pet.face
        await face.set_servo(pan_deg=event.pan_deg, tilt_deg=event.tilt_deg)
        await face.set_servo(mode="track")
        log.info("search: a face at pan %.1f after %.0f s", event.pan_deg, time.monotonic() - self.started)
        if await _wait_for_name(pet, self.person, self.cfg.search_naming_wait_s):
            return True
        newest = face.last_faces or event
        for f in newest.faces:
            self.checked.append(bearing_left_deg(f.cx, newest.pan_deg, self.cal) + self.turned_deg)
        log.info("search: not %s; on", self.who)
        for _ in iter(sub.get_nowait, None):    # frames from while it looked
            pass
        return False

    async def _pan_to(self, start: float, target: float, sub) -> Optional[FacesChanged]:
        """Pans from start to target; the first frame with a face not checked yet, if one comes."""
        speed = max(1.0, self.cfg.search_speed_deg_s)
        began = time.monotonic()
        span = abs(target - start)
        while True:
            for event in iter(sub.get_nowait, None):
                if event.faces and self._new_face(event) is not None:
                    return event
            done = min(span, speed * (time.monotonic() - began))
            await self.pet.face.set_servo(pan_deg=start + math.copysign(done, target - start))
            if done >= span:
                return None
            await asyncio.sleep(SEARCH_STEP_S)


async def search(pet: "App", person: Optional["Person"]) -> tuple[str, bool]:
    return await Search(pet, person).run()
