"""
Wheel odometry from the robot's Player server (position2d:0, ~50 Hz).

Found on the robot (2026-09-22): Player only streams while manual control
is armed through Valetudo; otherwise the subscription stays silent. That
is exactly when motion.py needs it (closed-loop turns and moves), so this
is not a general pose source. Valetudo's map pose is the one to use at
rest, and it can lag a whole manual session behind (the robot only saves
its map when remote control ends).

Samples are stamped with the PC's monotonic clock on receipt. Player's
own timestamp is the robot's uptime and isn't needed for control.
Yaw is unwrapped, so a turn past +-180 deg doesn't jump.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..config import PROJECT_ROOT

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OdomSample:
    t: float            # PC monotonic time of receipt
    x: float            # m, odometry frame (its origin is wherever it started)
    y: float
    yaw: float          # rad, counter-clockwise positive, unwrapped
    v: float            # m/s forward
    w: float            # rad/s, counter-clockwise positive
    stalled: bool


class PlayerOdometry:
    def __init__(self, host: str, port: int = 6665):
        self.host, self.port = host, port
        self._latest: Optional[OdomSample] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="player-odom", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def latest(self) -> Optional[OdomSample]:
        with self._lock:
            return self._latest

    def fresh(self, max_age_s: float = 0.3) -> Optional[OdomSample]:
        sample = self.latest()
        return sample if sample and time.monotonic() - sample.t <= max_age_s else None

    def _run(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "playerc-client"))
        import playerc_client as pc
        while not self._stop.is_set():
            try:
                client = pc.PlayerClient(self.host, self.port, timeout=3600)
                client.connect()
                client.subscribe(pc.PLAYER_POSITION2D_CODE, 0)
                last_raw: Optional[float] = None
                yaw = 0.0
                for pose in client.read_position2d():
                    if self._stop.is_set():
                        break
                    if last_raw is not None:
                        yaw += (pose.yaw - last_raw + math.pi) % (2 * math.pi) - math.pi
                    else:
                        yaw = pose.yaw
                    last_raw = pose.yaw
                    sample = OdomSample(time.monotonic(), pose.x, pose.y, yaw,
                                        pose.vx, pose.vyaw, pose.stalled)
                    with self._lock:
                        self._latest = sample
                client.close()
            except Exception as exc:  # noqa: BLE001 - reconnect whatever went wrong
                log.debug("player odometry: %s; reconnecting", exc)
                self._stop.wait(2.0)
