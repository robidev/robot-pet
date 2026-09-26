"""
Getting back onto the charger.

The robot finds its dock by sight: a pirouette to spot it, a straight run
at it, a turn, and backing on. From about a metre out that works (44-64 s
on 2026-09-23); from further, or out of sight, it searches in arcs and can
drift further away. So go_home first drives to a point straight out in front
of the dock, with Valetudo's own route planning, and starts the dock sequence
from there. If the robot isn't docked within dock_timeout_s, it's stopped
and sent round again, dock_attempts times in all.

The approach point is computed, not taught: `approach_cm` out from the
charger (Valetudo's charger_location) along the line through where the
robot's centre sits when it's docked, which is the way it backs in.

Both are learned while the robot is on the dock, and kept in the memory
database's kv table, because only then are they both right: once the robot
leaves, Valetudo moves charger_location to where the robot's centre was on
the dock (seen 2026-09-23: (2548, 2540) -> (2564, 2551), and back on
docking), which leaves no direction to go by. A live charger far from both
learned points means the dock has moved; then it docks from where it is,
and learns the new place on arrival.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Optional

from ..events import VacuumStateChanged
from ..io.vacuum import MapPose, VacuumState

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

DOCK_KEY = "dock.docked_and_charger"      # "x,y,charger_x,charger_y"
# The robot's centre sits ~20 cm out from the charger when docked. Closer is
# Valetudo mid-update: on docking it reports the charger at the robot's
# centre for a moment (2 cm away on 2026-09-23 13:30), which gave a point
# behind the dock. Further is a map glitch.
MIN_DOCKED_OFFSET_CM = 10.0
MAX_DOCKED_OFFSET_CM = 60.0
# A live charger further than this from both learned points: the dock moved.
DOCK_MOVED_CM = 40.0


def distance(a: MapPose, b: MapPose) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def approach_point(charger: Optional[MapPose], docked: Optional[MapPose],
                   distance_cm: float) -> Optional[MapPose]:
    """`distance_cm` straight out in front of the dock, or None if that can't be told."""
    if charger is None or docked is None:
        return None
    offset = distance(charger, docked)
    if offset < MIN_DOCKED_OFFSET_CM or offset > MAX_DOCKED_OFFSET_CM:
        return None
    return MapPose(charger.x + (docked.x - charger.x) / offset * distance_cm,
                   charger.y + (docked.y - charger.y) / offset * distance_cm)


async def await_arrival(vacuum, name: str, target: MapPose, arrive_cm: float,
                        timeout_s: float = 180.0, start_grace_s: float = 8.0,
                        poll_s: float = 1.0) -> tuple[str, bool]:
    """
    Valetudo reports a go_to through its status: moving, then idle (or an
    error). Idle is also how it ends when it stopped short: a goal on
    furniture stops it in front, one inside sends it round to the far side
    (2026-09-26). So where it ended up decides whether it arrived.
    """
    started_at = time.monotonic()
    started = False
    while time.monotonic() - started_at < timeout_s:
        state = await vacuum.refresh()
        if state.status == "error":
            return f"couldn't get to {name}: my base reported an error ({state.error})", False
        if state.moving:
            started = True
        elif started or time.monotonic() - started_at > start_grace_s:
            # The map pose lags the stop a little.
            await asyncio.sleep(2 * poll_s)
            return _arrival(await vacuum.refresh(), name, target, arrive_cm, started)
        await asyncio.sleep(poll_s)
    return f"gave up on getting to {name}: it took too long", False


def _arrival(state: VacuumState, name: str, target: MapPose, arrive_cm: float,
             started: bool) -> tuple[str, bool]:
    if state.pose is None:
        if started:
            return f"arrived at {name}, I think: I can't tell where I am", True
        return f"never set off for {name}, and I can't tell where I am", False
    off = distance(state.pose, target)
    if off <= arrive_cm:
        return (f"arrived at {name}" if started else f"already at {name}"), True
    if not started:
        return f"never set off for {name}: my base ignored me", False
    return f"stopped {off:.0f} cm from {name}: something may be in the way", False


class Dock:
    def __init__(self, pet: "App"):
        self.pet = pet
        self.cfg = pet.cfg.motion
        self.docked_pose: Optional[MapPose] = None
        self.charger: Optional[MapPose] = None     # as seen while docked
        # Pace of the status polling while going home (tests shorten these).
        self.poll_s = 1.0
        self.start_grace_s = 8.0
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        db = self.pet.db
        stored = db.kv_get(DOCK_KEY) if db is not None else None
        if stored:
            x, y, cx, cy = (float(v) for v in stored.split(","))
            self.docked_pose, self.charger = MapPose(x, y), MapPose(cx, cy)
        sub = self.pet.bus.subscribe(VacuumStateChanged)
        self._task = asyncio.create_task(self._watch(sub), name="dock")
        # The vacuum's first poll ran before this subscribed: on the dock at
        # start-up is only in its current state.
        self._remember_docked(self.pet.vacuum.state)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    async def _watch(self, sub) -> None:
        async for event in sub:
            self._remember_docked(event.state)

    def _remember_docked(self, state: VacuumState) -> None:
        """While docked, the robot's centre and the charger say which way the dock faces."""
        pose, charger = state.pose, state.charger
        if not state.docked or pose is None or charger is None:
            return
        if approach_point(charger, pose, self.cfg.dock_approach_cm) is None:
            return          # on top of each other, or implausibly far apart
        if (self.docked_pose is not None and self.charger is not None
                and distance(self.docked_pose, pose) < 2.0 and distance(self.charger, charger) < 2.0):
            return
        self.docked_pose, self.charger = MapPose(pose.x, pose.y), MapPose(charger.x, charger.y)
        point = self.approach_point()
        log.info("docked at (%.0f, %.0f), charger at (%.0f, %.0f): approach point (%.0f, %.0f)",
                 pose.x, pose.y, charger.x, charger.y, point.x, point.y)
        if self.pet.db is not None:
            self.pet.db.kv_set(DOCK_KEY, f"{pose.x:.1f},{pose.y:.1f},{charger.x:.1f},{charger.y:.1f}")

    def approach_point(self) -> Optional[MapPose]:
        return approach_point(self.charger, self.docked_pose, self.cfg.dock_approach_cm)

    def approach(self, state: VacuumState) -> Optional[MapPose]:
        """The approach point, unless the live map says the dock has moved since."""
        point = self.approach_point()
        live = state.charger
        if point is None or live is None:
            return point
        if min(distance(live, self.charger), distance(live, self.docked_pose)) > DOCK_MOVED_CM:
            log.info("going home: the dock seems to have moved to (%.0f, %.0f)", live.x, live.y)
            return None
        return point

    async def go_home(self) -> tuple[str, bool]:
        """The whole trip: approach point, dock sequence, retries. For App.start_motion."""
        vacuum = self.pet.vacuum
        if self.pet.motion is not None:
            await self.pet.motion.disarm()
        state = await vacuum.refresh()
        if state.docked:
            return "already on my dock", True
        for attempt in range(1, self.cfg.dock_attempts + 1):
            point = self.approach(state)
            if point is None:
                log.info("going home: no approach point known yet, docking from here")
            elif state.pose is not None and distance(state.pose, point) <= self.cfg.dock_near_cm:
                log.info("going home: already in front of the dock")
            else:
                log.info("going home (attempt %d): to (%.0f, %.0f) in front of the dock first",
                         attempt, point.x, point.y)
                await vacuum.go_to(point.x, point.y)
                outcome, ok = await await_arrival(vacuum, "the spot in front of my dock", point,
                                                  self.cfg.arrive_cm, start_grace_s=self.start_grace_s,
                                                  poll_s=self.poll_s)
                if not ok:
                    log.warning("going home: %s; docking from here", outcome)
            await vacuum.dock()
            state, docked = await self._await_docked(self.cfg.dock_timeout_s)
            if docked:
                return "back on my dock", True
            if state.status == "error":
                return f"couldn't dock: my base reported an error ({state.error})", False
            log.warning("going home: not docked after %.0f s (attempt %d of %d)",
                        self.cfg.dock_timeout_s, attempt, self.cfg.dock_attempts)
            await vacuum.stop_motion()
            if point is None:
                break           # without an approach point, a retry is the same search again
            state = await vacuum.refresh()
        return ("couldn't find my dock: I stopped searching so I don't wander off. "
                "Someone may need to put me back on it"), False

    async def _await_docked(self, timeout_s: float) -> tuple[VacuumState, bool]:
        started_at = time.monotonic()
        state = await self.pet.vacuum.refresh()
        while time.monotonic() - started_at < timeout_s:
            if state.docked:
                return state, True
            if state.status == "error":
                return state, False
            await asyncio.sleep(self.poll_s)
            state = await self.pet.vacuum.refresh()
        return state, state.docked
