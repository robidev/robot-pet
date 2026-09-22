"""
Closed-loop turns and moves (PLAN.md C4): Valetudo manual control is the
actuator, Player wheel odometry the feedback.

What the robot taught us (2026-09-22, scripts/calibrate_motion.py):

- Arming manual control spins the lidar up, and moves are ignored until it
  is ready, ~6 s later. With no moves for a few seconds it spins down, and
  the next move waits another 6 s. So a session is kept armed with
  zero-vectors (as Valetudo's own joystick does every 250 ms) while motions
  follow each other, and disarmed after `idle_disarm_s` without any.
- velocity 0.3 moves at ~12.6 cm/s; angle a spins in place at ~a deg/s,
  clockwise for positive a. It coasts a little after a stop, so each loop
  stops early by (current rate x coast_s) and then checks where it ended.
- Player odometry only streams while armed, which is exactly when it's
  needed. Valetudo's map pose can lag a whole session, so it isn't used.

The loops run at the 200 ms command rate. A stall (Player's flag), a
timeout or a missing odometry stream ends a motion early; cancelling the
task (a stop reflex) sends a zero vector on the way out.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..config import MotionConfig

if TYPE_CHECKING:
    from ..io.vacuum import VacuumAdapter

log = logging.getLogger(__name__)


class MotionError(Exception):
    """A motion that couldn't be done or finished; the text is for the brain."""


@dataclass(frozen=True)
class MotionResult:
    what: str               # "turn" | "move"
    asked: float            # degrees (CCW positive) or cm (forward positive)
    done: float             # what odometry says actually happened
    ok: bool
    reason: str = ""

    def describe(self) -> str:
        unit = "degrees" if self.what == "turn" else "cm"
        if self.ok:
            return f"{self.what} done: {self.done:+.0f} {unit} (asked {self.asked:+.0f})"
        return (f"{self.what} stopped early after {self.done:+.0f} of {self.asked:+.0f} {unit}: "
                f"{self.reason}")


class Motion:
    def __init__(self, vacuum: "VacuumAdapter", odometry, cfg: MotionConfig):
        self.vacuum = vacuum
        self.odometry = odometry
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._armed_at: Optional[float] = None
        self._last_motion_end = 0.0
        self._driving = False
        self._keepalive: Optional[asyncio.Task] = None
        self.last_disarmed_at: Optional[float] = None

    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    async def close(self) -> None:
        await self.disarm()
        close = getattr(self.odometry, "close", None)
        if close:
            close()

    # --- the armed session ------------------------------------------------------

    async def _ensure_armed(self) -> None:
        if self._armed_at is None:
            log.info("arming manual control (lidar spin-up, ~%.0fs)", self.cfg.warmup_s)
            start = getattr(self.odometry, "start", None)
            if start:
                start()
            await self.vacuum.manual_start()
            self._armed_at = time.monotonic()
            self._keepalive = asyncio.create_task(self._keep_alive(), name="motion-keepalive")
        wait = self._armed_at + self.cfg.warmup_s - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        deadline = time.monotonic() + 5.0
        while self._sample() is None:
            if time.monotonic() > deadline:
                raise MotionError("I can't feel my wheels (no odometry from the robot)")
            await asyncio.sleep(0.05)

    async def _keep_alive(self) -> None:
        """Zero-vectors while armed and idle, so the lidar stays up; disarms when unused."""
        try:
            while True:
                await asyncio.sleep(self.cfg.resend_s)
                if self._driving:
                    continue
                idle = time.monotonic() - max(self._last_motion_end, self._armed_at or 0.0)
                if not self._lock.locked() and idle > self.cfg.idle_disarm_s:
                    asyncio.create_task(self.disarm())
                    return
                try:
                    await self.vacuum.manual_move(0.0, 0.0)
                except Exception as exc:  # noqa: BLE001 - the next tick tries again
                    log.debug("keep-alive failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def disarm(self) -> None:
        task, self._keepalive = self._keepalive, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        if self._armed_at is None:
            return
        self._armed_at = None
        self.last_disarmed_at = time.time()
        try:
            await self.vacuum.manual_end()
        except Exception:  # noqa: BLE001 - nothing more to be done about it
            log.exception("disarming manual control failed")
        log.info("manual control disarmed")

    def _sample(self):
        fresh = getattr(self.odometry, "fresh", None)
        return fresh() if fresh else self.odometry.latest()

    async def _send(self, velocity: float, angle: float) -> None:
        """A move that may be lost: the next one is 200 ms behind it."""
        try:
            await self.vacuum.manual_move(velocity, angle)
        except Exception as exc:  # noqa: BLE001
            log.info("manual move lost (%s)", exc)

    async def _wait_for_odometry(self):
        """
        None when odometry stays silent for 3 s. The robot's WiFi stalls for
        over a second now and then; stopping and waiting it out beats either
        driving blind or giving up at the first gap.
        """
        sample = self._sample()
        if sample is not None:
            return sample
        await self._send(0.0, 0.0)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            sample = self._sample()
            if sample is not None:
                return sample
        return None

    # --- motions -------------------------------------------------------------------

    async def turn_by(self, degrees: float) -> MotionResult:
        """Turns in place; positive is counter-clockwise (to the robot's left)."""
        degrees = max(-self.cfg.max_turn_deg, min(self.cfg.max_turn_deg, degrees))
        async with self._lock:
            await self._ensure_armed()
            start = self._sample()
            target = start.yaw + math.radians(degrees)
            reason = ""
            try:
                self._driving = True
                deadline = time.monotonic() + self.cfg.turn_timeout_s
                for _ in range(3):              # a pass, then up to two corrections
                    reason = await self._turn_to(target, deadline)
                    if reason:
                        break
                    await self._stop_and_settle()
                    if abs(math.degrees(target - self.odometry.latest().yaw)) <= self.cfg.turn_tolerance_deg:
                        break
            finally:
                await self._halt()
            done = math.degrees(self.odometry.latest().yaw - start.yaw)
            ok = not reason and abs(done - degrees) <= self.cfg.turn_tolerance_deg * 2
            return MotionResult("turn", degrees, done, ok, reason or ("" if ok else "overshot"))

    async def _turn_to(self, target: float, deadline: float) -> str:
        c = self.cfg
        while True:
            s = await self._wait_for_odometry()
            if s is None:
                return "lost touch with my wheels"
            if s.stalled:
                return "something is blocking my wheels"
            if time.monotonic() > deadline:
                return "took too long"
            err = math.degrees(target - s.yaw)
            coast = abs(math.degrees(s.w)) * c.coast_s
            if abs(err) <= c.turn_tolerance_deg + coast:
                return ""
            rate = max(c.min_turn_rate_deg_s, min(c.max_turn_rate_deg_s, abs(err) * c.turn_gain))
            # Valetudo's angle turns clockwise; ours is counter-clockwise.
            angle = -math.copysign(rate, err) / c.deg_s_per_angle
            await self._send(0.0, max(-180.0, min(180.0, angle)))
            await asyncio.sleep(c.resend_s)

    async def move_by(self, cm: float) -> MotionResult:
        """Drives straight; positive is forward. The bumpers are the only obstacle check."""
        cm = max(-self.cfg.max_move_cm, min(self.cfg.max_move_cm, cm))
        c = self.cfg
        async with self._lock:
            await self._ensure_armed()
            start = self._sample()
            heading = (math.cos(start.yaw), math.sin(start.yaw))
            direction = 1.0 if cm >= 0 else -1.0

            def progress_cm() -> float:
                s = self.odometry.latest()
                return ((s.x - start.x) * heading[0] + (s.y - start.y) * heading[1]) * 100

            reason = ""
            deadline = time.monotonic() + abs(cm) / c.slow_cm_s + 5.0
            try:
                self._driving = True
                while True:
                    s = await self._wait_for_odometry()
                    if s is None:
                        reason = "lost touch with my wheels"
                        break
                    if s.stalled:
                        reason = "I bumped into something"
                        break
                    if time.monotonic() > deadline:
                        reason = "took too long"
                        break
                    remaining = abs(cm) - direction * progress_cm()
                    coast = abs(s.v) * 100 * c.coast_s
                    # Just the coast: stopping a tolerance early too left moves
                    # 2-3 cm short on the robot.
                    if remaining <= coast + 0.5:
                        break
                    speed = c.cruise_cm_s if remaining > 3 * c.cruise_cm_s * c.coast_s + 10 else c.slow_cm_s
                    await self._send(direction * speed / c.cm_s_per_velocity, 0.0)
                    await asyncio.sleep(c.resend_s)
                await self._stop_and_settle()
            finally:
                await self._halt()
            done = progress_cm()
            ok = not reason and abs(done - cm) <= max(c.move_tolerance_cm * 2, abs(cm) * 0.1)
            return MotionResult("move", cm, done, ok, reason or ("" if ok else "missed"))

    async def _stop_and_settle(self) -> None:
        await self._send(0.0, 0.0)
        await asyncio.sleep(self.cfg.settle_s)

    async def _halt(self) -> None:
        """Always runs, cancelled or not: a zero vector, and the motion is over."""
        self._driving = False
        self._last_motion_end = time.monotonic()
        try:
            await asyncio.shield(self.vacuum.manual_move(0.0, 0.0))
        except Exception:  # noqa: BLE001
            log.exception("could not send the stop vector")
