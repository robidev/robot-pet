"""
Low battery off the dock, whatever the initiative switch says (PLAN.md Step
7, E5a): it's safety, not initiative.

- At `motion.low_battery_tell_pct` (30%) the brain is told, once a
  discharge, to head home; it goes with go_home, which comes in via the
  front of the dock (spatial/dock.py).
- At `motion.low_battery_dock_pct` (22%) petd sends it home itself, cutting
  short whatever it was doing, and tells the brain.
- The robot has its own low-battery return (Valetudo: event.low_power_back,
  below 20% while cleaning; whether also during a go_to isn't known). It
  shows as a change of status to "returning" that petd didn't make (a pose
  update still "returning" just after petd's own trip ended is that trip's
  tail, not the robot's own). Then petd stands
  aside: its own motion is cancelled without a pause or stop to Valetudo,
  the motion tools refuse until it's docked, and the brain is told.
- A "stop" pauses a trip home like any motion. Below the tell level, petd
  sends it home again `motion.low_battery_resume_s` later unless it has
  docked by then; the brain hears of it with the stop.
- A go_to refused on a low battery (Valetudo's error 12) doesn't strand it:
  going home then docks from where it stands (spatial/dock.py).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from ..events import LowBattery, VacuumStateChanged

if TYPE_CHECKING:
    from ..app import App
    from ..io.vacuum import VacuumState

log = logging.getLogger(__name__)


class BatteryWatch:
    def __init__(self, pet: "App"):
        self.pet = pet
        self.cfg = pet.cfg.motion
        self.told = False               # the brain was told to go home, this discharge
        self.own_return = False         # the robot is taking itself home
        self._resume: Optional[asyncio.Task] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        sub = self.pet.bus.subscribe(VacuumStateChanged)
        self._task = asyncio.create_task(self._watch(sub), name="battery")
        await self.check(self.pet.vacuum.state)

    async def close(self) -> None:
        for task in (self._task, self._resume):
            if task is not None:
                task.cancel()

    def heading_home(self) -> bool:
        dock = self.pet.dock
        return self.own_return or (dock is not None and dock.going_home)

    async def _watch(self, sub) -> None:
        async for event in sub:
            try:
                await self.check(event.state, event.changed)
            except Exception:  # noqa: BLE001 - keep watching whatever happens
                log.exception("battery watch failed on %r", event)

    async def check(self, state: "VacuumState", changed: Optional[tuple] = None) -> None:
        """`changed`: the fields that changed; None (at start-up) counts as all."""
        cfg = self.cfg
        if state.docked:
            if self.own_return:
                log.info("battery: docked after the robot's own return")
            self.own_return = self.told = False
            self._cancel_resume()
            return
        dock = self.pet.dock
        if state.status == "returning" and not (dock is not None and dock.going_home):
            if not self.own_return and (changed is None or "status" in changed):
                self.own_return = True
                await self._stand_aside(state.battery_level)
            return
        if self.own_return:
            # Paused (a stop) or given up: no longer on its way by itself.
            log.info("battery: the robot's own return ended (%s)", state.status)
            self.own_return = False
        level = state.battery_level
        if level is None or not state.reachable or self.heading_home():
            return
        if level <= cfg.low_battery_dock_pct:
            log.warning("battery at %d%%: sending it home", level)
            self.told = True
            self.pet.bus.publish(LowBattery(level=level, action="docking"))
            await self.pet.go_home()
            self._tell(f"My battery is at {level}%, so I'm already driving back to my dock: "
                       "that comes before anything else now.")
        elif level <= cfg.low_battery_tell_pct and not self.told:
            self.told = True
            outings = self.pet.outings
            if outings is not None and outings.current is not None:
                return          # the outing's watchdog sends it home, and says why
            log.warning("battery at %d%%: telling the brain", level)
            self.pet.bus.publish(LowBattery(level=level, action="told"))
            self._tell(f"My battery is at {level}%. Head back to the dock now with go_home, "
                       f"whatever I'm doing; at {cfg.low_battery_dock_pct}% I'm sent home anyway.")

    async def _stand_aside(self, level: Optional[int]) -> None:
        log.warning("battery: the robot is taking itself home (battery %s%%); standing aside", level)
        self.pet.bus.publish(LowBattery(level=level if level is not None else -1, action="own_return"))
        await self.pet.cancel_motion()      # no pause or stop: that would halt its return
        self._tell(f"My base has started taking me back to my dock by itself (its own low-battery "
                   f"return; battery {level}%). I can't drive until I'm docked.")

    def stopping(self) -> None:
        """
        Called by App.stop_everything before it halts anything, while it can
        still tell whether a trip home is under way.
        """
        state = self.pet.vacuum.state
        level = state.battery_level
        if (level is None or state.docked or level > self.cfg.low_battery_tell_pct
                or not self.heading_home()):
            return
        self._cancel_resume()
        self._resume = asyncio.create_task(self._resume_later(level), name="battery-resume")
        if self.pet.brain is not None:
            self.pet.brain.note(f"That stop interrupted my trip home on a low battery ({level}%): "
                                f"in {self.cfg.low_battery_resume_s:.0f} s I head home again, "
                                "unless I'm docked by then.")

    async def _resume_later(self, level: int) -> None:
        await asyncio.sleep(self.cfg.low_battery_resume_s)
        self._resume = None
        state = self.pet.vacuum.state
        if state.docked or self.heading_home():
            return
        level = state.battery_level if state.battery_level is not None else level
        log.warning("battery at %d%%: heading home again after the stop", level)
        self.pet.bus.publish(LowBattery(level=level, action="resume"))
        await self.pet.go_home()
        self._tell(f"My battery is at {level}%, and the stop was a while ago: I'm driving back "
                   "to my dock again.")

    def _cancel_resume(self) -> None:
        task, self._resume = self._resume, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _tell(self, text: str) -> None:
        if self.pet.brain is not None:
            self.pet.brain.tell(text, kind="event")
