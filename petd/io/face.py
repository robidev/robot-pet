"""
Async adapter around face-api's FaceApiClient + FaceEventStream.

- /ws events are bridged onto the bus (FacesChanged, MotionDetected) from
  the stream's background thread via publish_threadsafe.
- FacesPresence is a debounced "someone is here / nobody is here".
- /api/status is polled for servo/eye state. Whenever the device reports
  its audio destination as unconfigured (e.g. after a reboot), the init
  sequence (detection on, the head's recognition off, audio destination,
  gain) is re-applied.
  A reboot is logged as a warning, with the board's reset reason.
- current_faces() prefers GET /api/face/current (fresh boxes) and falls
  back to the last pushed event on firmware without that endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..bus import EventBus
from ..config import FaceConfig
from ..events import FaceDeviceConnection, FacesChanged, FacesPresence, MotionDetected
from ..net import local_ip_towards

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Face:
    """One detected face, box normalized 0..1 to the camera frame."""
    id: int                     # enrolled id, -1 = not recognized
    confidence: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def recognized(self) -> bool:
        return self.id >= 0

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def w(self) -> float:
        return self.right - self.left

    @property
    def h(self) -> float:
        return self.bottom - self.top

    @classmethod
    def from_device(cls, raw: dict) -> "Face":
        box = raw.get("box", {})
        return cls(
            id=int(raw.get("id", -1)), confidence=float(raw.get("confidence", 0.0)),
            left=box.get("left", 0.0), top=box.get("top", 0.0),
            right=box.get("right", 0.0), bottom=box.get("bottom", 0.0),
        )


@dataclass(frozen=True)
class FaceFrame:
    faces: tuple
    pan_deg: float
    tilt_deg: float
    device_utc: Optional[float]
    age_ms: Optional[int]
    fresh: bool                 # False = fallback to the last pushed event (may be stale)
    received: float = field(default_factory=time.time)


@dataclass(frozen=True)
class FaceDeviceState:
    reachable: bool = False
    face_enabled: bool = False
    recognition: bool = False
    enrolled: int = 0
    audio_configured: bool = False
    servo_mode: Optional[str] = None
    pan_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    eye_x: Optional[float] = None
    eye_y: Optional[float] = None
    aperture: Optional[float] = None
    eye_mode: Optional[str] = None
    motion: bool = False
    time_synced: bool = False
    uptime_s: Optional[int] = None
    reset_reason: Optional[str] = None
    wifi_rssi: Optional[int] = None           # dBm (firmware from 2026-09-23 on)
    wifi_disconnects: Optional[int] = None    # since the board booted
    wifi_last_reason: Optional[int] = None
    updated: float = 0.0


def parse_status(status: dict) -> FaceDeviceState:
    eye = status.get("eye", {})
    servo = status.get("servo", {})
    wifi = status.get("wifi", {})
    return FaceDeviceState(
        reachable=True,
        face_enabled=bool(status.get("face_enabled")),
        recognition=bool(status.get("face_recognition")),
        enrolled=int(status.get("enrolled_faces", 0)),
        audio_configured=bool(status.get("audio_configured")),
        servo_mode=servo.get("mode"), pan_deg=servo.get("pan"), tilt_deg=servo.get("tilt"),
        eye_x=eye.get("x"), eye_y=eye.get("y"), aperture=eye.get("aperture"), eye_mode=eye.get("mode"),
        motion=bool(status.get("motion")),
        time_synced=bool(status.get("time", {}).get("synced")),
        uptime_s=status.get("uptime_s"), reset_reason=status.get("reset_reason"),
        wifi_rssi=wifi.get("rssi"), wifi_disconnects=wifi.get("disconnects"),
        wifi_last_reason=wifi.get("last_disconnect_reason"),
        updated=time.time(),
    )


# ESP-IDF wifi_err_reason_t, the ones a router or the radio tends to give.
WIFI_REASONS = {
    1: "unspecified", 2: "authentication expired", 3: "deauthenticated: leaving",
    4: "disassociated: inactivity", 8: "disassociated: leaving", 15: "key handshake timed out",
    200: "beacon timeout (lost the router)", 201: "router not found", 202: "authentication failed",
    203: "association failed", 204: "handshake timed out", 205: "connection failed",
}


def wifi_drop_message(state: FaceDeviceState) -> str:
    reason = state.wifi_last_reason
    return (f"last reason {reason} ({WIFI_REASONS.get(reason, 'see wifi_err_reason_t')}), "
            f"signal now {state.wifi_rssi} dBm, {state.wifi_disconnects} drops since it booted")


def reboot_detected(previous_uptime: Optional[int], uptime: Optional[int]) -> bool:
    """The board's uptime went backwards between two polls. Its own reboots
    (a panic, the watchdog, a brownout) are otherwise silent: it just comes
    back, and petd re-initializes it."""
    return previous_uptime is not None and uptime is not None and uptime < previous_uptime


class PresenceDebouncer:
    """
    Turns raw face-set changes into FacesPresence events: present=True as
    soon as any face shows up, present=False only after the set has stayed
    empty for `debounce_s`.
    """

    def __init__(self, bus: EventBus, debounce_s: float):
        self.bus = bus
        self.debounce_s = debounce_s
        self.present = False
        self._lost_timer: Optional[asyncio.TimerHandle] = None

    def update(self, faces: tuple) -> None:
        if faces:
            if self._lost_timer is not None:
                self._lost_timer.cancel()
                self._lost_timer = None
            if not self.present:
                self.present = True
                self.bus.publish(FacesPresence(present=True, faces=faces))
        elif self.present and self._lost_timer is None:
            loop = asyncio.get_running_loop()
            self._lost_timer = loop.call_later(self.debounce_s, self._lost)

    def _lost(self) -> None:
        self._lost_timer = None
        self.present = False
        self.bus.publish(FacesPresence(present=False))


class FaceAdapter(ABC):
    def __init__(self, bus: EventBus, cfg: FaceConfig):
        self.bus = bus
        self.cfg = cfg
        self._state = FaceDeviceState()
        self._last_faces: Optional[FacesChanged] = None
        self.presence = PresenceDebouncer(bus, cfg.faces_lost_debounce_s)
        self._sub = None

    @property
    def state(self) -> FaceDeviceState:
        return self._state

    @property
    def last_faces(self) -> Optional[FacesChanged]:
        return self._last_faces

    async def start(self) -> None:
        # Presence tracking listens to the bus so real and fake faces share it.
        self._sub = self.bus.subscribe(FacesChanged)
        self._presence_task = asyncio.create_task(self._track_presence(), name="face-presence")

    async def close(self) -> None:
        if self._sub is not None:
            self._sub.close()
            self._presence_task.cancel()

    async def _track_presence(self) -> None:
        async for event in self._sub:
            self._last_faces = event
            self.presence.update(event.faces)

    @abstractmethod
    async def current_faces(self) -> FaceFrame: ...
    @abstractmethod
    async def snapshot(self): ...
    @abstractmethod
    async def set_servo(self, *, mode: Optional[str] = None, pan_deg: Optional[float] = None,
                        tilt_deg: Optional[float] = None) -> None: ...
    @abstractmethod
    async def set_eye(self, x: float, y: float = 0.0, aperture: float = 1.0) -> None: ...
    @abstractmethod
    async def set_eye_mode(self, mode: str) -> None: ...
    @abstractmethod
    async def enroll_next_face(self) -> None:
        """Arms enrollment: the first face in the next frame gets a new id (then disarms)."""
    @abstractmethod
    async def cancel_enroll(self) -> None: ...
    @abstractmethod
    async def list_enrolled(self) -> list: ...
    @abstractmethod
    async def delete_enrolled(self, face_id: int) -> int: ...


class RobotFace(FaceAdapter):
    def __init__(self, cfg: FaceConfig, bus: EventBus, pc_ip: str = "auto"):
        super().__init__(bus, cfg)
        from face_client import FaceApiClient, FaceEventStream
        self._client = FaceApiClient(cfg.host, cfg.port, timeout=5.0)
        self._status_client = FaceApiClient(cfg.host, cfg.port, timeout=3.0)
        self._events = FaceEventStream(self._client)
        self._pc_ip = pc_ip
        self._poll_task: Optional[asyncio.Task] = None
        self._has_current_endpoint: Optional[bool] = None
        self._last_init_attempt = float("-inf")
        # Survives unreachable polls, which is what a reboot looks like from here.
        self._last_uptime: Optional[int] = None
        self._last_wifi_drops: Optional[int] = None
        self._last_answer = float("-inf")       # time.monotonic() of the last good poll

    async def start(self) -> None:
        await super().start()
        from face_client import parse_utc

        def on_face(evt) -> None:
            faces = tuple(Face.from_device(f) for f in evt.faces)
            self.bus.publish_threadsafe(FacesChanged(
                faces=faces, pan_deg=evt.servo.pan_deg, tilt_deg=evt.servo.tilt_deg,
                device_utc=parse_utc(evt.utc), seq=evt.seq))

        def on_motion(evt) -> None:
            self.bus.publish_threadsafe(MotionDetected(
                active=evt.active, pan_deg=evt.servo.pan_deg, tilt_deg=evt.servo.tilt_deg,
                device_utc=parse_utc(evt.utc)))

        def on_error(exc: Exception) -> None:
            log.debug("face event stream: %s", exc)

        self._events.on_face = on_face
        self._events.on_motion = on_motion
        self._events.on_error = on_error
        self._events.start()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="face-status")

    async def close(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
        await asyncio.to_thread(self._events.stop)
        await super().close()

    async def _poll_loop(self) -> None:
        while True:
            try:
                status = await asyncio.to_thread(self._status_client.get_status)
            except Exception as exc:  # noqa: BLE001 - any failure means unreachable
                silent_s = time.monotonic() - self._last_answer
                if silent_s < self.cfg.offline_after_s:
                    # A WiFi stall: keep the last state rather than tell everyone it's gone.
                    log.debug("face didn't answer (%s); %.0f s since it last did", exc, silent_s)
                else:
                    if self._state.reachable:
                        log.warning("face unreachable for %.0f s: %s", silent_s, exc)
                        self.bus.publish(FaceDeviceConnection(connected=False, detail=str(exc)))
                    self._state = FaceDeviceState(reachable=False, updated=time.time())
            else:
                self._last_answer = time.monotonic()
                was_reachable = self._state.reachable
                self._state = parse_status(status)
                if reboot_detected(self._last_uptime, self._state.uptime_s):
                    log.warning("face board rebooted (reset reason: %s, up %s s)",
                                self._state.reset_reason, self._state.uptime_s)
                if self._state.uptime_s is not None:
                    self._last_uptime = self._state.uptime_s
                drops = self._state.wifi_disconnects
                if drops is not None:
                    if self._last_wifi_drops is None:
                        log.info("face WiFi: %s", wifi_drop_message(self._state))
                    elif drops > self._last_wifi_drops:
                        # The board reconnects by itself now; this says how often, and why.
                        log.warning("face WiFi dropped and came back: %s", wifi_drop_message(self._state))
                    self._last_wifi_drops = drops
                if not was_reachable:
                    self.bus.publish(FaceDeviceConnection(connected=True))
                needs_init = not self._state.audio_configured or not self._state.face_enabled
                if needs_init and time.monotonic() - self._last_init_attempt > 5.0:
                    self._last_init_attempt = time.monotonic()
                    await self._initialize()
            await asyncio.sleep(self.cfg.status_poll_s)

    async def _initialize(self) -> None:
        """(Re)applies the settings the pet depends on; the device forgets some on reboot."""
        pc_ip = self._pc_ip
        if pc_ip == "auto":
            pc_ip = local_ip_towards(self.cfg.host, self.cfg.port)
        log.info("initializing face: detection on, audio -> %s:%d", pc_ip, self.cfg.audio_port)
        try:
            await asyncio.to_thread(self._client.set_face_detection, True)
            # Recognition happens on the PC (memory/recognition.py); on the head it
            # only cost the tracking loop time.
            await asyncio.to_thread(self._client.set_recognition, False)
            await asyncio.to_thread(self._client.set_audio_destination, pc_ip, self.cfg.audio_port)
            if self.cfg.audio_gain is not None:
                await asyncio.to_thread(self._client.set_audio_gain, self.cfg.audio_gain)
        except Exception as exc:  # noqa: BLE001 - retried on the next poll
            log.warning("face init failed (will retry): %s", exc)

    async def current_faces(self) -> FaceFrame:
        from face_client import FaceApiHTTPError, parse_utc
        if self._has_current_endpoint is not False:
            try:
                cur = await asyncio.to_thread(self._client.get_current_faces)
            except FaceApiHTTPError as exc:
                if exc.status_code != 404:
                    raise
                log.warning("firmware has no /api/face/current; using last pushed face event (may be stale)")
                self._has_current_endpoint = False
            else:
                self._has_current_endpoint = True
                return FaceFrame(
                    faces=tuple(Face.from_device(f) for f in cur.faces),
                    pan_deg=cur.servo.pan_deg, tilt_deg=cur.servo.tilt_deg,
                    device_utc=parse_utc(cur.utc), age_ms=cur.age_ms, fresh=True)
        last = self._last_faces
        if last is None:
            return FaceFrame((), self._state.pan_deg or 90.0, self._state.tilt_deg or 90.0,
                             None, None, fresh=False)
        return FaceFrame(last.faces, last.pan_deg, last.tilt_deg, last.device_utc,
                         int((time.time() - last.t) * 1000), fresh=False)

    async def snapshot(self):
        """face_client.Snapshot (jpeg bytes + servo pose + capture time)."""
        return await asyncio.to_thread(self._client.get_snapshot)

    async def set_servo(self, *, mode=None, pan_deg=None, tilt_deg=None) -> None:
        await asyncio.to_thread(self._client.set_servo, mode=mode, pan_deg=pan_deg, tilt_deg=tilt_deg)

    async def set_eye(self, x: float, y: float = 0.0, aperture: float = 1.0) -> None:
        await asyncio.to_thread(self._client.set_eye_target, x, y, aperture)

    async def set_eye_mode(self, mode: str) -> None:
        await asyncio.to_thread(self._client.set_eye_mode, mode)

    async def enroll_next_face(self) -> None:
        await asyncio.to_thread(self._client.enroll_face, True)

    async def cancel_enroll(self) -> None:
        await asyncio.to_thread(self._client.enroll_face, False)

    async def list_enrolled(self) -> list:
        return await asyncio.to_thread(self._client.list_enrolled_faces)

    async def delete_enrolled(self, face_id: int) -> int:
        return await asyncio.to_thread(self._client.delete_enrolled_face, face_id)


_TINY_JPEG: Optional[bytes] = None


def _tiny_jpeg() -> bytes:
    global _TINY_JPEG
    if _TINY_JPEG is None:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buf, "JPEG")
        _TINY_JPEG = buf.getvalue()
    return _TINY_JPEG


class FakeFace(FaceAdapter):
    """In-memory stand-in. Tests/--fake mode inject faces with show()."""

    def __init__(self, cfg: FaceConfig, bus: EventBus):
        super().__init__(bus, cfg)
        self.commands: list = []
        self.enrolled: list = []
        self._frame_faces: tuple = ()
        self._state = FaceDeviceState(reachable=True, face_enabled=True, recognition=True,
                                      audio_configured=True, servo_mode="track",
                                      pan_deg=90.0, tilt_deg=90.0, eye_mode="auto",
                                      time_synced=True, updated=time.time())

    def show(self, *faces: Face) -> None:
        """Simulates the device pushing a face-set change."""
        self._frame_faces = tuple(faces)
        self.bus.publish(FacesChanged(faces=self._frame_faces, pan_deg=self._state.pan_deg,
                                      tilt_deg=self._state.tilt_deg, device_utc=time.time()))

    async def current_faces(self) -> FaceFrame:
        return FaceFrame(self._frame_faces, self._state.pan_deg, self._state.tilt_deg,
                         time.time(), 0, fresh=True)

    async def snapshot(self):
        """A 1x1 black JPEG: something for the recognizer's (fake) engine to look at."""
        self.commands.append(("snapshot",))
        from face_client import Snapshot
        return Snapshot(jpeg=_tiny_jpeg(), utc=None, age_ms=0,
                        servo_pan_deg=self._state.pan_deg, servo_tilt_deg=self._state.tilt_deg)

    async def set_servo(self, *, mode=None, pan_deg=None, tilt_deg=None) -> None:
        self.commands.append(("servo", mode, pan_deg, tilt_deg))
        from dataclasses import replace
        self._state = replace(self._state,
                              servo_mode=mode or self._state.servo_mode,
                              pan_deg=self._state.pan_deg if pan_deg is None else pan_deg,
                              tilt_deg=self._state.tilt_deg if tilt_deg is None else tilt_deg)

    async def set_eye(self, x: float, y: float = 0.0, aperture: float = 1.0) -> None:
        self.commands.append(("eye", x, y, aperture))

    async def set_eye_mode(self, mode: str) -> None:
        self.commands.append(("eye_mode", mode))

    async def enroll_next_face(self) -> None:
        """Like the device: enrolls whoever is in view (already known or not)."""
        self.commands.append(("enroll",))
        if self._frame_faces and len(self.enrolled) < 7:
            new_id = max(self.enrolled, default=-1) + 1
            self.enrolled.append(new_id)

    async def cancel_enroll(self) -> None:
        self.commands.append(("enroll_cancel",))

    async def list_enrolled(self) -> list:
        return list(self.enrolled)

    async def delete_enrolled(self, face_id: int) -> int:
        self.enrolled.remove(face_id)
        return len(self.enrolled)
