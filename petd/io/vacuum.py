"""
Async adapter around vacuum-api's ValetudoClient.

- One background poller fetches map + state attributes every
  poll_interval_s and publishes VacuumStateChanged when anything but the
  timestamp changes. Everything else reads `state` instead of hitting
  Valetudo (get_map() is the whole map JSON).
- Commands run in worker threads, serialized by a lock, on a separate
  HTTP session from the poller.
- drive() is reimplemented as an async loop (instead of the client's
  blocking drive()) so a stop can cancel it mid-way.

State-changing calls are NOT exercised by tests against the real robot.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..bus import EventBus
from ..config import VacuumConfig
from ..events import VacuumStateChanged

log = logging.getLogger(__name__)

MOVING_STATUSES = ("moving", "manual_control", "cleaning", "returning")


@dataclass(frozen=True)
class MapPose:
    x: float                    # Valetudo map units (cm)
    y: float
    angle: Optional[float] = None  # degrees, Valetudo convention (see PLAN.md 4.1 calibration)


@dataclass(frozen=True)
class VacuumState:
    reachable: bool = False
    status: Optional[str] = None     # docked | idle | returning | cleaning | paused | manual_control | moving | error
    flag: Optional[str] = None
    error: Optional[dict] = None
    battery_level: Optional[int] = None
    battery_flag: Optional[str] = None
    pose: Optional[MapPose] = None
    charger: Optional[MapPose] = None
    pixel_size: Optional[int] = None
    updated: float = 0.0             # PC wall clock of the poll that produced this

    @property
    def docked(self) -> bool:
        return self.status == "docked"

    @property
    def moving(self) -> bool:
        return self.status in MOVING_STATUSES


def parse_state(map_json: Optional[dict], attributes: Optional[list]) -> dict:
    """Valetudo map JSON + attributes list -> VacuumState field values."""
    out: dict = {}
    for attr in attributes or []:
        cls = attr.get("__class")
        if cls == "StatusStateAttribute":
            out.update(status=attr.get("value"), flag=attr.get("flag"), error=attr.get("error"))
        elif cls == "BatteryStateAttribute":
            out.update(battery_level=attr.get("level"), battery_flag=attr.get("flag"))
    if map_json:
        out["pixel_size"] = map_json.get("pixelSize")
        for entity in map_json.get("entities", []):
            points = entity.get("points") or []
            if len(points) < 2:
                continue
            if entity.get("type") == "robot_position":
                out["pose"] = MapPose(points[0], points[1], (entity.get("metaData") or {}).get("angle"))
            elif entity.get("type") == "charger_location":
                out["charger"] = MapPose(points[0], points[1])
    return out


def log_changes(old: VacuumState, new: VacuumState, changed: tuple) -> None:
    """The run log's account of the base: status, errors, the dock's position,
    and (at DEBUG) the robot's position whenever it's off the dock, so a trip
    that went wrong can be followed afterwards."""
    if "reachable" in changed and new.reachable:
        log.info("vacuum reachable")        # going unreachable is logged with its cause
    if "status" in changed:
        log.info("vacuum %s -> %s (battery %s%%, at %s)",
                 old.status, new.status, new.battery_level, _where(new.pose))
    if "error" in changed and new.error:
        log.warning("vacuum error: %s", new.error)
    if "charger" in changed:
        log.info("vacuum's dock at %s (was %s)", _where(new.charger), _where(old.charger))
    if "pose" in changed and not new.docked:
        log.debug("vacuum at %s (%s)", _where(new.pose), new.status)


def _where(pose: Optional[MapPose]) -> str:
    if pose is None:
        return "?"
    heading = "" if pose.angle is None else f", heading {pose.angle:.0f}"
    return f"({pose.x:.0f}, {pose.y:.0f}{heading})"


def changed_fields(old: VacuumState, new: VacuumState) -> tuple:
    return tuple(
        f.name for f in dataclasses.fields(VacuumState)
        if f.name != "updated" and getattr(old, f.name) != getattr(new, f.name)
    )


class VacuumAdapter(ABC):
    def __init__(self, bus: EventBus):
        self.bus = bus
        self._state = VacuumState()
        self._map: Optional[dict] = None

    @property
    def state(self) -> VacuumState:
        return self._state

    @property
    def last_map(self) -> Optional[dict]:
        """Most recently polled raw map JSON (may be a couple of seconds old)."""
        return self._map

    def _set_state(self, new: VacuumState) -> None:
        old, self._state = self._state, new
        changed = changed_fields(old, new)
        if changed:
            log_changes(old, new, changed)
            self.bus.publish(VacuumStateChanged(state=new, changed=changed))

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    @abstractmethod
    async def go_to(self, x: float, y: float) -> None: ...
    @abstractmethod
    async def dock(self) -> None: ...
    @abstractmethod
    async def pause(self) -> None: ...
    @abstractmethod
    async def stop_motion(self) -> None: ...
    @abstractmethod
    async def drive(self, velocity: float, angle: float, duration_s: float) -> None:
        """MOVES THE ROBOT. velocity -1..1, angle -180..180 (Valetudo steering angle)."""
    @abstractmethod
    async def refresh(self) -> VacuumState: ...

    # Raw manual control, for spatial/motion.py. On this robot (Roborock V1)
    # arming spins the lidar up, and moves are ignored for ~6 s until it's
    # ready; with no moves for a few seconds it spins down again. velocity
    # 0.3 is ~12.6 cm/s; angle a spins in place at ~a deg/s, clockwise for
    # positive a (Valetudo sends omega = -a in rad/s).
    @abstractmethod
    async def manual_start(self) -> None: ...
    @abstractmethod
    async def manual_move(self, velocity: float, angle: float) -> None:
        """MOVES THE ROBOT (after the warm-up)."""
    @abstractmethod
    async def manual_end(self) -> None: ...


class ValetudoVacuum(VacuumAdapter):
    def __init__(self, cfg: VacuumConfig, bus: EventBus):
        super().__init__(bus)
        from valetudo_client import ValetudoClient
        self.cfg = cfg
        self._last_answer = float("-inf")    # time.monotonic() of the last good poll
        self._poll_client = ValetudoClient(cfg.host, cfg.port, timeout=5.0)
        self._cmd_client = ValetudoClient(cfg.host, cfg.port, timeout=5.0)
        # Manual moves are resent every 200 ms: one stuck behind a WiFi stall
        # is worth dropping, not waiting 5 s for.
        self._move_client = ValetudoClient(cfg.host, cfg.port, timeout=1.0)
        self._cmd_lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        self._drive_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.refresh()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="vacuum-poll")

    async def close(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
        await self._cancel_drive()

    async def refresh(self) -> VacuumState:
        try:
            map_json, attrs = await asyncio.to_thread(self._fetch)
        except Exception as exc:  # noqa: BLE001 - network errors of any kind mean "unreachable"
            silent_s = time.monotonic() - self._last_answer
            if silent_s < self.cfg.offline_after_s:
                # A WiFi stall: keep the last state rather than tell everyone it's gone.
                log.debug("vacuum didn't answer (%s); %.0f s since it last did", exc, silent_s)
                return self._state
            if self._state.reachable:
                log.warning("vacuum unreachable for %.0f s: %s", silent_s, exc)
            self._set_state(dataclasses.replace(self._state, reachable=False, updated=time.time()))
            return self._state
        self._last_answer = time.monotonic()
        self._map = map_json
        self._set_state(VacuumState(reachable=True, updated=time.time(), **parse_state(map_json, attrs)))
        return self._state

    def _fetch(self) -> tuple:
        return self._poll_client.get_map(), self._poll_client.get_state_attributes()

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.poll_interval_s)
            await self.refresh()

    async def _command(self, fn, *args) -> None:
        async with self._cmd_lock:
            await asyncio.to_thread(fn, *args)
        # Reflect the command's effect without waiting a full poll interval.
        asyncio.create_task(self.refresh())

    async def go_to(self, x: float, y: float) -> None:
        await self._cancel_drive()
        await self._command(self._cmd_client.go_to, x, y)

    async def dock(self) -> None:
        await self._cancel_drive()
        await self._command(self._cmd_client.go_to_dock)

    async def pause(self) -> None:
        await self._cancel_drive()
        await self._command(self._cmd_client.pause_cleaning)

    async def stop_motion(self) -> None:
        """
        Best-effort halt of whatever is moving: cancels a running drive()
        (which disables manual control on the way out), then sends
        BasicControl stop if Valetudo still reports movement.
        """
        await self._cancel_drive()
        state = await self.refresh()
        if state.moving:
            await self._command(self._cmd_client.stop_cleaning)

    async def drive(self, velocity: float, angle: float, duration_s: float) -> None:
        await self._cancel_drive()
        self._drive_task = asyncio.create_task(self._drive(velocity, angle, duration_s), name="vacuum-drive")
        await self._drive_task

    async def _drive(self, velocity: float, angle: float, duration_s: float) -> None:
        client = self._cmd_client
        async with self._cmd_lock:
            await asyncio.to_thread(client.enable_manual_control)
            try:
                deadline = time.monotonic() + duration_s
                while time.monotonic() < deadline:
                    await asyncio.to_thread(client.drive_vector, velocity, angle)
                    await asyncio.sleep(self.cfg.drive_update_interval_s)
            finally:
                # Must run even when cancelled, or the robot keeps its last vector
                # until Valetudo's own timeout.
                await asyncio.shield(asyncio.to_thread(client.disable_manual_control))

    async def manual_start(self) -> None:
        await self._cancel_drive()
        await self._manual(self._cmd_client.enable_manual_control, give_up_after_s=10.0)

    async def manual_move(self, velocity: float, angle: float) -> None:
        await self._manual(self._move_client.drive_vector, velocity, angle, give_up_after_s=0.0)

    async def manual_end(self) -> None:
        # Left armed, the lidar keeps spinning and the robot ignores go_to:
        # keep trying through a WiFi stall.
        await asyncio.shield(self._manual(self._cmd_client.disable_manual_control,
                                          give_up_after_s=20.0))
        asyncio.create_task(self.refresh())

    async def _manual(self, fn, *args, give_up_after_s: float) -> None:
        """Retries until `give_up_after_s` has passed (0 = a single retry at most,
        for a keep-alive connection Valetudo dropped while idle)."""
        import requests
        deadline = time.monotonic() + give_up_after_s
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._cmd_lock:
                    await asyncio.to_thread(fn, *args)
                return
            except requests.RequestException as exc:
                if attempt >= 2 and time.monotonic() >= deadline:
                    raise
                log.debug("manual control call failed (%s); retrying", exc)
                if attempt >= 2:
                    await asyncio.sleep(0.5)

    async def _cancel_drive(self) -> None:
        task, self._drive_task = self._drive_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class FakeVacuum(VacuumAdapter):
    """
    In-memory stand-in: records commands, "arrives" at go_to targets
    instantly, and in manual control integrates an odometry pose from the
    commanded speeds (after the same warm-up as the real robot, shortened).
    """

    CM_S_PER_VELOCITY = 42.0        # 0.3 -> 12.6 cm/s, as measured
    COAST_S = 0.3                   # it keeps going after a stop (82 deg for 67 asked, at 45 deg/s)

    def __init__(self, bus: EventBus, warmup_s: float = 0.0):
        super().__init__(bus)
        self.commands: list = []
        self.warmup_s = warmup_s
        self.odometry = FakeOdometry()
        self._manual_since: Optional[float] = None
        self._set_state(VacuumState(
            reachable=True, status="docked", flag="none", battery_level=100,
            battery_flag="charged", pose=MapPose(2560, 2549, 342),
            charger=MapPose(2560, 2530), pixel_size=5, updated=time.time(),
        ))

    def _update(self, **changes) -> None:
        self._set_state(dataclasses.replace(self._state, updated=time.time(), **changes))

    async def refresh(self) -> VacuumState:
        return self._state

    async def go_to(self, x: float, y: float) -> None:
        self.commands.append(("go_to", x, y))
        self._update(status="idle", pose=MapPose(x, y, self._state.pose.angle if self._state.pose else None))

    async def dock(self) -> None:
        self.commands.append(("dock",))
        self._update(status="docked", pose=self._state.charger)

    async def pause(self) -> None:
        self.commands.append(("pause",))
        self._update(status="paused")

    async def stop_motion(self) -> None:
        self.commands.append(("stop",))
        if self._state.moving:
            self._update(status="idle")

    async def manual_start(self) -> None:
        self.commands.append(("manual_start",))
        self._manual_since = time.monotonic()
        self._update(status="manual_control")

    async def manual_move(self, velocity: float, angle: float) -> None:
        self.commands.append(("manual_move", velocity, angle))
        if self._manual_since is None or time.monotonic() - self._manual_since < self.warmup_s:
            return                  # the lidar isn't up yet: ignored, like the real one
        self.odometry.command(velocity * self.CM_S_PER_VELOCITY / 100.0,
                              -angle * 3.141592653589793 / 180.0, self.COAST_S)

    async def manual_end(self) -> None:
        self.commands.append(("manual_end",))
        self._manual_since = None
        self.odometry.command(0.0, 0.0, 0.0)
        self._update(status="idle")

    async def drive(self, velocity: float, angle: float, duration_s: float) -> None:
        self.commands.append(("drive", velocity, angle, duration_s))
        self._update(status="manual_control")
        try:
            await asyncio.sleep(duration_s)
        finally:
            self._update(status="idle")


class FakeOdometry:
    """Integrates commanded speeds into a pose; `latest()` like PlayerOdometry."""

    def __init__(self):
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = 0.0
        self.stalled = False
        self._t = time.monotonic()
        self._stop_at: Optional[float] = None

    def _advance(self) -> None:
        import math
        now = time.monotonic()
        end = now if self._stop_at is None else min(now, self._stop_at)
        dt = max(0.0, end - self._t)
        self.yaw += self.w * dt
        self.x += self.v * dt * math.cos(self.yaw)
        self.y += self.v * dt * math.sin(self.yaw)
        self._t = now
        if self._stop_at is not None and now >= self._stop_at:
            self.v = self.w = 0.0
            self._stop_at = None

    def command(self, v: float, w: float, coast_s: float) -> None:
        self._advance()
        if v == 0.0 and w == 0.0 and (self.v or self.w):
            self._stop_at = time.monotonic() + coast_s
        else:
            self.v, self.w, self._stop_at = v, w, None

    def latest(self):
        from ..spatial.odometry import OdomSample
        self._advance()
        return OdomSample(t=time.monotonic(), x=self.x, y=self.y, yaw=self.yaw,
                          v=self.v, w=self.w, stalled=self.stalled)
