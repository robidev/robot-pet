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


class ValetudoVacuum(VacuumAdapter):
    def __init__(self, cfg: VacuumConfig, bus: EventBus):
        super().__init__(bus)
        from valetudo_client import ValetudoClient
        self.cfg = cfg
        self._poll_client = ValetudoClient(cfg.host, cfg.port, timeout=5.0)
        self._cmd_client = ValetudoClient(cfg.host, cfg.port, timeout=5.0)
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
            if self._state.reachable:
                log.warning("vacuum unreachable: %s", exc)
            self._set_state(dataclasses.replace(self._state, reachable=False, updated=time.time()))
            return self._state
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

    async def _cancel_drive(self) -> None:
        task, self._drive_task = self._drive_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class FakeVacuum(VacuumAdapter):
    """In-memory stand-in: records commands, "arrives" at go_to targets instantly."""

    def __init__(self, bus: EventBus):
        super().__init__(bus)
        self.commands: list = []
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

    async def drive(self, velocity: float, angle: float, duration_s: float) -> None:
        self.commands.append(("drive", velocity, angle, duration_s))
        self._update(status="manual_control")
        try:
            await asyncio.sleep(duration_s)
        finally:
            self._update(status="idle")
