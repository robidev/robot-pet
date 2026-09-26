"""
One fixed frame for what the pet remembers on the map (PLAN.md 4.2, C1).

The robot's map comes in a new frame now and then (~74 deg off once, on
2026-09-26): its origin is wherever the robot was when the map began. So
named places, and where people were seen, are kept in the frame of a
reference map (`motion.reference_map`, a Valetudo map JSON), and turned into
the current map's coordinates just before they're used.

- Checked when needed, not on every poll: if the current map's walls still
  land on the reference's under the last alignment (`frame_min_score`), it
  holds; otherwise mapgeo.align() runs again (~0.7 s, in a thread).
- No reference yet: the first map of at least `reference_min_m2` becomes it
  (a partial map would make a poor one). Until then, and whenever the
  current map can't be aligned, nothing can be placed.
- `--fake` has no map, and uses the robot's coordinates as they are.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from ..config import MotionConfig
from .mapgeo import Alignment, Grid, align

log = logging.getLogger(__name__)

IDENTITY = Alignment(0.0, 0.0, 0.0)


class MapFrame:
    def __init__(self, vacuum, cfg: MotionConfig, path: Path, fixed: bool = False):
        self.vacuum = vacuum
        self.cfg = cfg
        self.path = path
        self.fixed = fixed                          # --fake: no map, no frames
        self.reference: Optional[Grid] = None
        self.current: Optional[Alignment] = None    # current map -> reference
        if not fixed and path.exists():
            self.reference = Grid.from_valetudo(json.loads(path.read_text()))

    async def alignment(self) -> Optional[Alignment]:
        """Current map -> reference frame, or None if it can't be told."""
        if self.fixed:
            return IDENTITY
        map_json = self.vacuum.last_map
        if not map_json:
            return None
        grid = Grid.from_valetudo(map_json)
        if self.reference is None:
            return self._adopt(map_json, grid)
        if self.current is not None and self.reference.wall_hits(grid, self.current) >= self.cfg.frame_min_score:
            return self.current
        found = await asyncio.to_thread(align, self.reference, grid)
        if found.score < self.cfg.frame_min_score:
            log.warning("map frame: the current map doesn't match the reference (best %.2f)", found.score)
            self.current = None
            return None
        if self.current is None or found != self.current:
            log.info("map frame: current map is %.1f deg, (%.0f, %.0f) cm from the reference (score %.2f)",
                     found.angle_deg, found.tx, found.ty, found.score)
        self.current = found
        return found

    async def to_reference(self, x: float, y: float) -> Optional[tuple[float, float]]:
        a = await self.alignment()
        return a.apply(x, y) if a is not None else None

    async def to_current(self, x: float, y: float) -> Optional[tuple[float, float]]:
        a = await self.alignment()
        return a.inverse().apply(x, y) if a is not None else None

    def to_reference_now(self, x: float, y: float) -> Optional[tuple[float, float]]:
        """Like to_reference, with the last alignment found (no map check)."""
        a = IDENTITY if self.fixed else self.current
        return a.apply(x, y) if a is not None else None

    def _adopt(self, map_json: dict, grid: Grid) -> Optional[Alignment]:
        area_m2 = float(grid.floor.sum()) * grid.pixel_size ** 2 / 1e4
        if area_m2 < self.cfg.reference_min_m2:
            log.info("map frame: no reference yet, and %.0f m2 is too little of the room for one", area_m2)
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(map_json))
        self.reference, self.current = grid, IDENTITY
        log.info("map frame: this map (%.0f m2) is the reference now: %s", area_m2, self.path)
        return IDENTITY
