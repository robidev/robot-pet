"""
Nudges: the pet acting by itself (PLAN.md Step 7, E5c), all behind the
initiative switch (behavior/initiative.py).

Every `TICK_S`, with the switch on, at most one turn of kind "initiative"
goes to the brain, which may take it up or do nothing:
- **quiet hours beginning and ending**: one event each (goodnight: go home if
  out; good morning). Not when an outing was just ended for quiet hours:
  the watchdog already said so.
- **social**: someone in view for a couple of minutes without talking to it,
  when `why_not_remark` allows (rate, "be quiet", not mid-conversation).
- **restless**: nothing to do for `restless_after_min`, when `why_not_roam`
  allows: it may go exploring (behavior/explore.py) or stay put.

The brain must be idle for social and restless (no turn running or queued,
no conversation window open, not moving).

Two local reactions, no brain turn (like face tracking), only with the
switch on, outside quiet hours, with nobody in view and the head free:
- **perking up** at the PIR: the sensor is on the camera board, so it sees
  what the head faces; something moved there and no face is found: a
  curious eye and a glance up, where a standing adult's face would be.
- **idle glances**: a look left, right or up every few minutes, so it doesn't
  sit there like furniture.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING, Optional

from ..brain.tags import Action
from ..events import FacesPresence, MotionDetected, TurnEnded, TurnStarted

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

TICK_S = 30.0
SOCIAL_IN_VIEW_S = 120.0        # in view this long before the pet speaks up unasked
PERK_EVERY_S = 60.0             # a perk at the PIR at most this often
OUTING_JUST_ENDED_S = 120.0


class Nudges:
    def __init__(self, pet: "App"):
        self.pet = pet
        self.cfg = pet.cfg.initiative
        self.tick_s = TICK_S
        self.rng = random.Random()
        now = pet.initiative.clock()
        self.last_active_at = now               # a turn, a motion, petd starting
        self.last_heard_at: Optional[float] = None
        self.in_view_since: Optional[float] = None
        self.last_glance_at = now
        self.last_perk_at = float("-inf")
        self.last_nudge: Optional[tuple[float, str]] = None
        self._quiet: Optional[bool] = None       # quiet hours at the last tick
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._quiet = self.pet.initiative.quiet_hours()
        sub = self.pet.bus.subscribe(TurnStarted, TurnEnded, FacesPresence, MotionDetected)
        self._tasks = [asyncio.create_task(self._watch(sub), name="nudges-watch"),
                       asyncio.create_task(self._ticks(), name="nudges")]

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()

    @property
    def now(self) -> float:
        return self.pet.initiative.clock()

    async def _watch(self, sub) -> None:
        async for event in sub:
            try:
                if isinstance(event, TurnStarted) and event.kind == "heard":
                    self.last_heard_at = self.now
                elif isinstance(event, TurnEnded):
                    self.last_active_at = self.now
                elif isinstance(event, FacesPresence):
                    self.in_view_since = self.now if event.present else None
                elif isinstance(event, MotionDetected) and event.active:
                    self.perk()
            except Exception:  # noqa: BLE001 - keep watching whatever happens
                log.exception("nudges failed on %r", event)

    async def _ticks(self) -> None:
        while True:
            await asyncio.sleep(self.tick_s)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 - try again next tick
                log.exception("nudge tick failed")

    # --- one tick ------------------------------------------------------------------

    async def tick(self) -> Optional[str]:
        """Sends at most one nudge; returns which, for the tests and the log."""
        initiative, now = self.pet.initiative, self.now
        if self._busy():
            self.last_active_at = now
        quiet, was_quiet = initiative.quiet_hours(now), self._quiet
        self._quiet = quiet
        if not initiative.enabled:
            return None
        if was_quiet is not None and quiet != was_quiet:
            return self._quiet_hours_turned(quiet)
        if self._busy():
            return None
        if self._social_due(now):
            initiative.remarked(now)
            return self._nudge("social", self._social_text(now))
        if self._restless_due(now):
            if self._someone_in_view():
                initiative.remarked(now)
            return self._nudge("restless", self._restless_text(now))
        self._maybe_glance(now)
        return None

    def _busy(self) -> bool:
        pet = self.pet
        brain, listener = pet.brain, pet.listener
        if brain is not None and (getattr(brain, "busy", False) or not getattr(brain, "idle", True)):
            return True
        if listener is not None and listener.window_open(time.time()):
            return True
        return pet.moving or (pet.vacuum is not None and pet.vacuum.state.moving)

    def _someone_in_view(self) -> bool:
        face = self.pet.face
        return face is not None and face.presence.present

    def _social_due(self, now: float) -> bool:
        if self.pet.initiative.why_not_remark(now) is not None:
            return False
        if self.in_view_since is None or now - self.in_view_since < SOCIAL_IN_VIEW_S:
            return False
        quiet_for = self.cfg.remark_every_min * 60
        return self.last_heard_at is None or now - self.last_heard_at >= quiet_for

    def _restless_due(self, now: float) -> bool:
        outings = self.pet.outings
        if outings is not None and outings.current is not None:
            return False
        if self.pet.initiative.why_not_roam(now) is not None:
            return False
        return now - self.last_active_at >= self.cfg.restless_after_min * 60

    def _quiet_hours_turned(self, quiet: bool) -> Optional[str]:
        start, end = self.cfg.quiet_hours
        if quiet:
            outings = self.pet.outings
            ended = getattr(outings, "last_end", None)
            if ended is not None and self.now - ended[0] < OUTING_JUST_ENDED_S and ended[1].startswith("quiet"):
                return None     # the watchdog has just said so, and sent it home
            docked = self.pet.vacuum is None or self.pet.vacuum.state.docked
            return self._nudge("goodnight", (
                f"Quiet hours have begun ({start} until {end}): no wandering about until then, "
                "though I still answer when spoken to. "
                + ("" if docked else "I'm not on my dock: go home now (go_home). ")
                + "Say goodnight only if someone is in view."))
        return self._nudge("morning", (
            f"Quiet hours are over ({end}): I may wander about and speak up again. "
            "Say good morning only if someone is in view."))

    def _nudge(self, kind: str, text: str) -> str:
        log.info("nudge: %s", kind)
        self.last_nudge = (self.now, kind)
        self.last_active_at = self.now
        if self.pet.brain is not None:
            self.pet.brain.tell(text, kind="initiative")
        return kind

    # --- what a nudge says ------------------------------------------------------------

    def _who(self) -> str:
        people = self.pet.people
        if people is not None and self.pet.recognition_on:
            names, strangers = people.who_is_here()
            if strangers:
                names = names + (["someone I don't recognise"] if strangers == 1
                                 else [f"{strangers} people I don't recognise"])
            if names:
                return " and ".join(names)
        return "someone"

    def _social_text(self, now: float) -> str:
        minutes = max(2, round((now - self.in_view_since) / 60))
        return (f"{self._who()} has been in view for about {minutes} minutes without talking to "
                "me. I may say something (a remark, a question, something I noticed), or leave "
                "them be: saying nothing is fine.")

    def _restless_text(self, now: float) -> str:
        idle_min = round((now - self.last_active_at) / 60)
        seen = self.pet.initiative.seen_ago_s(now)
        if self._someone_in_view():
            around = f"{self._who()} is in view."
        else:
            around = (f"Nobody is in view (someone was, {round((seen or 0) / 60)} minutes ago), so "
                      "there's nobody to talk to.")
        return (f"Nothing has needed me for {idle_min} minutes. {around} If I feel like it I can "
                "go exploring (explore): a few spots, a look around at each, note anything worth "
                "remembering (note_observation), then go home. Or stay put.")

    # --- local: perking up and idle glances ---------------------------------------------

    def _head_free(self) -> Optional[object]:
        """The expressions, if the head may move by itself now."""
        pet = self.pet
        initiative = pet.initiative
        if not initiative.enabled or initiative.quiet_hours(self.now):
            return None
        if self._someone_in_view() or self._busy():
            return None
        expressions = getattr(pet.brain, "expressions", None)
        if expressions is None or expressions.head_busy or not pet.face.state.reachable:
            return None
        return expressions

    def perk(self) -> None:
        """Something moved in front of the head, and no face: a curious look, up a little."""
        now = self.now
        if now - self.last_perk_at < PERK_EVERY_S:
            return
        expressions = self._head_free()
        if expressions is None:
            return
        self.last_perk_at = self.last_glance_at = now
        log.info("perks up at motion")
        expressions.fire(Action("emote", "curious"))
        expressions.fire(Action("look", "up"))

    def _maybe_glance(self, now: float) -> None:
        if now - self.last_glance_at < self.cfg.idle_glance_every_min * 60:
            return
        expressions = self._head_free()
        if expressions is None:
            return
        self.last_glance_at = now
        direction = self.rng.choice(("left", "right", "up"))
        log.debug("idle glance %s", direction)
        expressions.fire(Action("look", direction))

    def status(self) -> dict:
        now = self.now
        last = self.last_nudge
        return {"idle_s": round(now - self.last_active_at),
                "last": None if last is None else {"kind": last[1], "ago_s": round(now - last[0])}}
