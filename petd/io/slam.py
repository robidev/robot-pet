"""
The robot's own SLAM pose, heading included (PLAN.md Step 5, 2026-10-11).

Valetudo gives the V1 no heading: it reports the direction of the last bit
of path, which was 130° off after a go_to that ended with a turn on the
spot, and anything after driving by remote. The robot's SLAM writes its
pose to /run/shm/SLAM_fprintf.log several times a second while the lidar
spins:

    8682.625 estimate 0.008 0.090 -1.581        uptime s, x m, y m, theta rad

in the frame of Valetudo's map: cm = 2560 + 100 x, 2560 - 100 y (y flipped),
and theta counter-clockwise from +x, so Valetudo's angle (clockwise from up)
is 90 - theta. Docked it read (2561, 2550) at 180.6°, as Valetudo, and on
2026-10-11 it agreed with every straight move to within a degree or two
(spatial/heading.py's sums of measured turns were 22° off after four).

Lines come only while the lidar spins, and the robot doesn't move without
it, so the last estimate holds while it stands still. This follows the log
with `ssh … tail -F` (the key as for io/robot_wifi.py), and starts over
after the robot reboots (the log is on tmpfs) or the link drops.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .robot_wifi import ssh_argv

if TYPE_CHECKING:
    from ..config import VacuumConfig
    from .vacuum import MapPose

log = logging.getLogger(__name__)

SLAM_LOG = "/run/shm/SLAM_fprintf.log"
ORIGIN_CM = 2560.0          # Valetudo's map coordinates of the SLAM origin
RETRY_S = 10.0


@dataclass(frozen=True)
class SlamPose:
    x: float                # Valetudo map cm
    y: float
    angle: float            # degrees, Valetudo's convention
    received: float         # time.time() it arrived


def parse(line: str, received: float = 0.0) -> Optional[SlamPose]:
    parts = line.split()
    if len(parts) != 5 or parts[1] != "estimate":
        return None
    try:
        x, y, theta = float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    return SlamPose(ORIGIN_CM + 100.0 * x, ORIGIN_CM - 100.0 * y,
                    (90.0 - math.degrees(theta)) % 360.0, received)


class SlamFollower:
    def __init__(self, cfg: "VacuumConfig"):
        self.cfg = cfg
        self.latest: Optional[SlamPose] = None
        self._task: Optional[asyncio.Task] = None
        self._proc: Optional[asyncio.subprocess.Process] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="slam-pose")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
        self._kill()

    def _kill(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()

    def pose(self) -> Optional["MapPose"]:
        from .vacuum import MapPose
        p = self.latest
        return MapPose(p.x, p.y, p.angle) if p is not None else None

    async def _run(self) -> None:
        argv = ssh_argv(self.cfg, f"tail -n 200 -F {SLAM_LOG} 2>/dev/null")
        while True:
            try:
                await self._follow(argv)
            except asyncio.CancelledError:
                self._kill()
                raise
            except Exception:  # noqa: BLE001 - ssh missing, key unreadable, ...
                log.exception("SLAM pose: following the robot's log failed")
            self.latest = None              # a reboot starts a new log; don't keep the old pose
            await asyncio.sleep(RETRY_S)

    async def _follow(self, argv: list) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        log.info("SLAM pose: following the robot's log")
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                pose = parse(line.decode(errors="replace"), time.time())
                if pose is not None:
                    self.latest = pose
        finally:
            self._kill()
            await self._proc.wait()
        log.info("SLAM pose: the robot's log stopped (exit %s); again in %.0f s",
                 self._proc.returncode, RETRY_S)
