"""
The initiative switch and the home rules (PLAN.md Step 7, E5a).

Everything the pet starts by itself asks here first: outings, unprompted
remarks, greetings and stranger notes. With the switch off it only answers:
what's said to it, and any tool a person asks for, still work. Low battery
is safety, not initiative (behavior/battery.py), and runs either way.

- The switch starts as `initiative.enabled` (off by default) and the
  dashboard changes it (`POST /initiative`) until petd stops.
- The rules are functions of the time and the pet's state, each answering
  with why not (a sentence for the brain and the dashboard) or None:
  `why_not_roam` (switch, quiet hours, someone seen lately, outings this
  hour, the battery) and `why_not_remark` (switch, quiet hours, someone in
  view, not mid-conversation, the rate and a "be quiet").
- Turns the pet starts by itself reach the brain as kind="initiative";
  motion tools called during one check `why_not_roam` (brain/tools.py).
  What a person asks for is never held back by these rules.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

from ..config import ConfigError, InitiativeConfig
from ..events import FacesPresence, InitiativeChanged

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)


def minutes_of_day(hhmm: str) -> int:
    try:
        hours, minutes = (int(part) for part in str(hhmm).split(":"))
    except ValueError:
        raise ConfigError(f"initiative.quiet_hours: {hhmm!r} isn't HH:MM") from None
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ConfigError(f"initiative.quiet_hours: {hhmm!r} isn't a time of day")
    return hours * 60 + minutes


def in_quiet_hours(now: float, quiet_hours: list) -> bool:
    """Whether `now` (epoch s) falls in [start, end) on the local clock; it may wrap past midnight."""
    start, end = (minutes_of_day(t) for t in quiet_hours)
    local = time.localtime(now)
    minute = local.tm_hour * 60 + local.tm_min
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


class Initiative:
    def __init__(self, pet: "App", clock: Callable[[], float] = time.time):
        self.pet = pet
        self.cfg: InitiativeConfig = pet.cfg.initiative
        if len(self.cfg.quiet_hours) != 2:
            raise ConfigError("initiative.quiet_hours: give a start and an end, e.g. [\"20:00\", \"09:00\"]")
        for hhmm in self.cfg.quiet_hours:
            minutes_of_day(hhmm)
        self.clock = clock
        self.enabled = bool(self.cfg.enabled)
        self.last_seen_at: Optional[float] = None   # a face, anyone's, last in view
        self.outings: list[float] = []               # when each outing began
        self.last_remark_at: Optional[float] = None
        self.muted_until = 0.0
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        log.info("initiative %s (config)", "on" if self.enabled else "off")
        self.pet.bus.publish(InitiativeChanged(on=self.enabled, source="config"))
        sub = self.pet.bus.subscribe(FacesPresence)
        self._task = asyncio.create_task(self._watch(sub), name="initiative")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    async def _watch(self, sub) -> None:
        async for event in sub:
            # Arriving and leaving both mark the time: in view until then.
            self.last_seen_at = event.t

    # --- the switch ---------------------------------------------------------------

    def set(self, on: bool, source: str = "api") -> None:
        on = bool(on)
        if on == self.enabled:
            return
        self.enabled = on
        log.warning("initiative switched %s (%s)", "on" if on else "off", source)
        self.pet.bus.publish(InitiativeChanged(on=on, source=source))
        if self.pet.brain is not None:
            self.pet.brain.note("Your initiative was switched " + (
                "on: you may go exploring and speak up by yourself again, within the house rules."
                if on else
                "off: until it's on again you only answer, and don't go anywhere or speak up by "
                "yourself."))

    # --- what the rules say ------------------------------------------------------

    def quiet_hours(self, now: Optional[float] = None) -> bool:
        return in_quiet_hours(self.clock() if now is None else now, self.cfg.quiet_hours)

    def seen_ago_s(self, now: Optional[float] = None) -> Optional[float]:
        """How long since a face was last in view: 0 while one is."""
        now = self.clock() if now is None else now
        face = self.pet.face
        if face is not None and face.presence.present:
            return 0.0
        return None if self.last_seen_at is None else max(0.0, now - self.last_seen_at)

    def outings_last_hour(self, now: Optional[float] = None) -> int:
        now = self.clock() if now is None else now
        return sum(1 for t in self.outings if now - t < 3600)

    def why_not_roam(self, now: Optional[float] = None) -> Optional[str]:
        """Why the pet may not go out on its own now; None when it may."""
        now = self.clock() if now is None else now
        cfg = self.cfg
        if not self.enabled:
            return "my initiative is switched off: I only go where I'm asked"
        if self.quiet_hours(now):
            return f"it's quiet hours ({cfg.quiet_hours[0]}-{cfg.quiet_hours[1]}): no wandering about"
        seen = self.seen_ago_s(now)
        if seen is None or seen > cfg.someone_seen_min * 60:
            return f"nobody has been around for {cfg.someone_seen_min:.0f} minutes"
        if self.outings_last_hour(now) >= cfg.max_outings_per_hour:
            return f"I've been out {cfg.max_outings_per_hour} times this hour already"
        vacuum, mcfg = self.pet.vacuum, self.pet.cfg.motion
        if vacuum is not None:
            state = vacuum.state
            level = state.battery_level
            if level is not None and state.docked and level < mcfg.min_battery_to_leave:
                return f"my battery is at {level}%, too low to leave the dock"
            if level is not None and not state.docked and level <= mcfg.low_battery_tell_pct:
                return f"my battery is at {level}%: I should be heading home"
        return None

    def why_not_remark(self, now: Optional[float] = None) -> Optional[str]:
        """Why the pet may not speak up unprompted now; None when it may."""
        now = self.clock() if now is None else now
        cfg = self.cfg
        if not self.enabled:
            return "initiative is off"
        if self.quiet_hours(now):
            return "quiet hours"
        if now < self.muted_until:
            return f"told to be quiet ({(self.muted_until - now) / 60:.0f} min left)"
        face = self.pet.face
        if face is None or not face.presence.present:
            return "nobody in view"
        brain, listener = self.pet.brain, self.pet.listener
        if (brain is not None and getattr(brain, "busy", False)) or (
                listener is not None and listener.window_open(now)):
            return "mid-conversation"
        if self.last_remark_at is not None and now - self.last_remark_at < cfg.remark_every_min * 60:
            return "spoke up recently"
        return None

    @property
    def greets(self) -> bool:
        """Greetings and stranger notes are the pet speaking first: only with the switch on."""
        return self.enabled

    # --- what happened -----------------------------------------------------------

    def outing_started(self, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self.outings = [t for t in self.outings if now - t < 3600] + [now]

    def remarked(self, now: Optional[float] = None) -> None:
        self.last_remark_at = self.clock() if now is None else now

    def mute(self, now: Optional[float] = None) -> None:
        """'Be quiet': no unprompted remarks for a while."""
        now = self.clock() if now is None else now
        self.muted_until = now + self.cfg.quiet_mutes_min * 60

    def status(self) -> dict:
        now = self.clock()
        seen = self.seen_ago_s(now)
        return {
            "on": self.enabled,
            "quiet_hours": self.cfg.quiet_hours,
            "in_quiet_hours": self.quiet_hours(now),
            "someone_seen_ago_s": None if seen is None else round(seen),
            "outings_last_hour": self.outings_last_hour(now),
            "roam": self.why_not_roam(now),
            "remark": self.why_not_remark(now),
        }
