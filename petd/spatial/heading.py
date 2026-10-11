"""
The robot's heading, kept right across turns on the spot (PLAN.md Step 5).

Valetudo gives the V1 no real heading: it reports the direction of the last
stretch of path. A turn on the spot doesn't change it, and backing up flips
it 180°; only driving forward (a go_to) sets it right, and docked it's taken
as straight away from the dock (io/vacuum.py). So after petd's own manual
motions (spatial/motion.py) the heading is the one before them minus the
turn the wheels measured, until Valetudo drives again: a go_to, docking or
cleaning makes its own heading good once more.

Angles are Valetudo's: degrees, growing clockwise on the map.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

from ..events import VacuumStateChanged

if TYPE_CHECKING:
    from ..bus import EventBus
    from ..io.vacuum import MapPose, VacuumAdapter

log = logging.getLogger(__name__)

# Valetudo driving on its own: its heading is the direction it drove.
VALETUDO_DRIVES = ("moving", "returning", "cleaning", "docked")


class Heading:
    def __init__(self, vacuum: "VacuumAdapter", bus: "EventBus"):
        self.vacuum = vacuum
        self.bus = bus
        self.override: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        sub = self.bus.subscribe(VacuumStateChanged)
        self._task = asyncio.create_task(self._watch(sub), name="heading")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    async def _watch(self, sub) -> None:
        async for event in sub:
            if self.override is not None and event.state.status in VALETUDO_DRIVES:
                log.debug("heading: Valetudo drives (%s); its heading again", event.state.status)
                self.override = None

    def angle(self) -> Optional[float]:
        if self.override is not None:
            return self.override
        pose = self.vacuum.state.pose
        return pose.angle if pose is not None else None

    def pose(self) -> Optional["MapPose"]:
        """Valetudo's pose with this heading."""
        pose = self.vacuum.state.pose
        if pose is None or self.override is None:
            return pose
        return dataclasses.replace(pose, angle=self.override)

    def turned(self, before: Optional[float], left_deg: float) -> None:
        """A manual motion turned the robot left_deg (counter-clockwise) from `before`."""
        if before is None:
            return
        self.override = (before - left_deg) % 360.0
        log.debug("heading: %.0f after turning %+.0f deg", self.override, left_deg)
