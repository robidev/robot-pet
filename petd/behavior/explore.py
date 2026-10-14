"""
Outings: the pet out on its own initiative (PLAN.md Step 7, E5b).

- The `explore` tool (brain/tools.py) drives to a random spot of open floor
  (`pick_spot`: anywhere on the map, decision 3, clear of anything the
  lidar has seen, a hop away from where it is and from the outing's earlier
  spots). Arriving is told to the brain, which may look around, explore on
  or go home.
- Called in a turn the pet started itself, it is an outing: up to
  `initiative.spots_per_outing` spots, counted against
  `max_outings_per_hour` from the first. Asked by a person, it is just a
  trip somewhere.
- While an outing is on, a watchdog sends the pet home and tells the brain
  why when the initiative is switched off, quiet hours begin, nobody has
  been seen for `someone_seen_min`, or the battery reaches
  `motion.low_battery_tell_pct`. The robot taking itself home ends it
  quietly; so does being docked again.
- A person taking over (any motion tool outside a self-started turn: "come
  here" mid-outing) ends the outing where it stands: what a person asks
  for isn't the pet wandering about.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..app import App
    from ..spatial.mapgeo import Grid

log = logging.getLogger(__name__)

CHECK_EVERY_S = 2.0
RECENT_SPOTS = 3            # a person's "go explore" avoids the last few spots too


def pick_spot(grid: "Grid", robot_xy: tuple[float, float], avoid: list[tuple[float, float]],
              clearance_cm: float, min_hop_cm: float, rng: Optional[random.Random] = None,
              tries: int = 400) -> Optional[tuple[float, float]]:
    """
    A random point of mapped floor at least `clearance_cm` from any wall
    pixel, and at least `min_hop_cm` from the robot and from every point in
    `avoid`; None if `tries` samples find none. Map cm.
    """
    rng = rng or random.Random()
    cells = np.argwhere(grid.floor)
    if not len(cells):
        return None
    away_from = [robot_xy, *avoid]
    for _ in range(tries):
        row, col = cells[rng.randrange(len(cells))]
        x = (col + grid.x0 + 0.5) * grid.pixel_size
        y = (row + grid.y0 + 0.5) * grid.pixel_size
        if any(math.hypot(x - ax, y - ay) < min_hop_cm for ax, ay in away_from):
            continue
        if grid.is_free(x, y, clearance_cm):
            return x, y
    return None


@dataclass
class Outing:
    started_at: float
    spots: list = field(default_factory=list)       # map cm, in the order visited
    left_dock: bool = False


class Outings:
    def __init__(self, pet: "App"):
        self.pet = pet
        self.current: Optional[Outing] = None
        self.recent_spots: list[tuple[float, float]] = []
        self.check_every_s = CHECK_EVERY_S
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._watch(), name="outings")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    def begin(self) -> Outing:
        initiative = self.pet.initiative
        now = initiative.clock()
        initiative.outing_started(now)
        self.current = Outing(started_at=now, left_dock=not self.pet.vacuum.state.docked)
        log.info("outing begins")
        return self.current

    def visited(self, spot: tuple[float, float], outing: Optional[Outing]) -> None:
        if outing is not None:
            outing.spots.append(spot)
        self.recent_spots = (self.recent_spots + [spot])[-RECENT_SPOTS:]

    def person_took_over(self) -> None:
        """A person asked for a motion: the outing is over, wherever the pet is."""
        if self.current is not None:
            log.info("outing ends: a person asked for something")
            self.current = None

    def status(self) -> Optional[dict]:
        outing = self.current
        if outing is None:
            return None
        return {"spots": len(outing.spots), "of": self.pet.cfg.initiative.spots_per_outing,
                "for_s": round(self.pet.initiative.clock() - outing.started_at)}

    # --- the watchdog --------------------------------------------------------------

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self.check_every_s)
            try:
                await self.check()
            except Exception:  # noqa: BLE001 - keep watching whatever happens
                log.exception("outing watchdog failed")

    async def check(self) -> None:
        outing = self.current
        if outing is None:
            return
        state = self.pet.vacuum.state
        if not state.docked:
            outing.left_dock = True
        elif outing.left_dock:
            log.info("outing over: back on the dock")
            self.current = None
            return
        battery = self.pet.battery
        if battery is not None and battery.own_return:
            log.info("outing over: the robot is taking itself home")
            self.current = None
            return
        why = self.why_end()
        if why is None:
            return
        log.warning("outing over: %s", why)
        self.current = None
        if self.pet.dock is not None and self.pet.dock.going_home:
            return
        await self.pet.go_home()
        if self.pet.brain is not None:
            self.pet.brain.tell(f"My outing is over: {why}. I'm already driving back to my dock.",
                                kind="event")

    def why_end(self) -> Optional[str]:
        """Why an outing must end now; None while it may go on."""
        initiative = self.pet.initiative
        cfg = initiative.cfg
        now = initiative.clock()
        if not initiative.enabled:
            return "my initiative was switched off"
        if initiative.quiet_hours(now):
            return f"quiet hours have begun ({cfg.quiet_hours[0]})"
        seen = initiative.seen_ago_s(now)
        if seen is None or seen > cfg.someone_seen_min * 60:
            return f"nobody has been around for {cfg.someone_seen_min:.0f} minutes"
        level = self.pet.vacuum.state.battery_level
        if level is not None and level <= self.pet.cfg.motion.low_battery_tell_pct:
            return f"my battery is down to {level}%"
        return None
