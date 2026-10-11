"""
The robot's own SLAM pose, heading included (PLAN.md Step 5, 2026-10-11).

Valetudo gives the V1 no heading: it reports the direction of the last bit
of path, which was 130° off after a go_to that ended with a turn on the
spot, and anything after driving by remote. Player has none either (only
wheel odometry, F1). The robot's SLAM writes its pose to
/run/shm/SLAM_fprintf.log several times a second while the lidar spins:

    8682.625 estimate 0.008 0.090 -1.581        uptime s, x m, y m, theta rad

in the frame of Valetudo's map: cm = 2560 + 100 x, 2560 - 100 y (y flipped),
and theta counter-clockwise from +x, so Valetudo's angle (clockwise from up)
is 90 - theta. Docked it read (2561, 2550) at 180.6°, as Valetudo, and on
2026-10-11 it agreed with every straight move to within a degree or two
(spatial/heading.py's sums of measured turns were 22° off after four).

On the robot a socat (like the speaker's on 6000) answers each connection
to `vacuum.slam_port` with the newest estimate line in the log's last 4 KB,
and closes. petd asks every `slam_poll_s`: nothing on the robot keeps state,
and a WiFi stall costs one answer (the last pose is kept). Lines come only
while the lidar spins and the robot doesn't move without it, so the last
estimate holds while it stands still, also across a reboot.

petd starts the socat over SSH (the key as for io/robot_wifi.py) when the
port refuses: after the robot's nightly reboot, or the first time. Nothing
in the robot's boot path changes (Robin, 2026-10-11).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .robot_wifi import run_process, ssh_argv

if TYPE_CHECKING:
    from ..config import VacuumConfig
    from .vacuum import MapPose

log = logging.getLogger(__name__)

SLAM_LOG = "/run/shm/SLAM_fprintf.log"
ORIGIN_CM = 2560.0          # Valetudo's map coordinates of the SLAM origin
ASK_TIMEOUT_S = 2.0
START_EVERY_S = 30.0        # at most this often, the socat started over SSH


def responder_command(port: int) -> str:
    """Starts the socat on the robot unless it runs; prints "running" if it does then."""
    match = f"'^socat TCP-LISTEN:{port}'"
    # Only the socat goes to the background: backgrounding `a || b` kept ssh
    # waiting on the subshell for as long as the socat ran.
    return (f"pgrep -f {match} >/dev/null || (setsid nohup socat TCP-LISTEN:{port},reuseaddr,fork "
            f"SYSTEM:'tail -c 4096 {SLAM_LOG} | grep estimate | tail -n 1' "
            f"</dev/null >/dev/null 2>&1 &); sleep 0.5; pgrep -f {match} >/dev/null && echo running")


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


class SlamPoller:
    def __init__(self, cfg: "VacuumConfig", run=None):
        self.cfg = cfg
        self.run = run or run_process
        self.latest: Optional[SlamPose] = None
        self._task: Optional[asyncio.Task] = None
        self._started_at = float("-inf")

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="slam-pose")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    def pose(self) -> Optional["MapPose"]:
        from .vacuum import MapPose
        p = self.latest
        return MapPose(p.x, p.y, p.angle) if p is not None else None

    async def _loop(self) -> None:
        while True:
            try:
                await self.poll()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the next poll tries again
                log.exception("SLAM pose: poll failed")
            await asyncio.sleep(self.cfg.slam_poll_s)

    async def poll(self) -> None:
        try:
            line = await self.ask()
        except ConnectionRefusedError:
            await self.start_responder()
            return
        except (OSError, asyncio.TimeoutError) as exc:
            log.debug("SLAM pose: no answer (%s); keeping the last", exc)
            return
        pose = parse(line, time.time())
        if pose is not None:
            self.latest = pose

    async def ask(self) -> str:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.cfg.host, self.cfg.slam_port), ASK_TIMEOUT_S)
        try:
            line = await asyncio.wait_for(reader.readline(), ASK_TIMEOUT_S)
        finally:
            writer.close()
        return line.decode(errors="replace")

    async def start_responder(self) -> None:
        if time.monotonic() - self._started_at < START_EVERY_S or not self.cfg.ssh_key:
            return
        self._started_at = time.monotonic()
        code, out = await self.run(ssh_argv(self.cfg, responder_command(self.cfg.slam_port)), 20.0)
        if code == 0 and out.endswith("running"):
            log.info("SLAM pose: started the robot's responder on port %d", self.cfg.slam_port)
        else:
            log.warning("SLAM pose: couldn't start the robot's responder (exit %s): %s",
                        code, out or "no output")
