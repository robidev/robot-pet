"""
Python client for the LilyGo-Cam-RobotFace HTTP + WebSocket API.

Built directly against the actual endpoints in ~/LilyGo-Cam-RobotFace's
src/control_server.cpp (not a generic spec) -- this is the "python HTTP API
call to face" section of ~/robot-pet/todo.txt's low-level-glue-logic MVP.
See face-api/README.md for where this deviates from that original wishlist.

Example:
    from face_client import FaceApiClient, FaceEventStream

    face = FaceApiClient("192.168.4.1")
    face.get_status()                     # full JSON status
    face.set_face_detection(True)
    face.set_recognition(True)
    snap = face.get_snapshot()            # Snapshot(jpeg=..., servo_pan_deg=..., ...)

    events = FaceEventStream(face)
    events.on_face = lambda evt: print("face:", evt)
    events.on_motion = lambda evt: print("motion:", evt)
    events.start()
    ...
    events.stop()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None


class FaceApiError(Exception):
    """Base error for anything going wrong talking to the robot face."""


class FaceApiHTTPError(FaceApiError):
    """The device responded with a non-2xx status code."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


# Settings accepted by GET /api/camera (control_server.cpp's camera_handler).
CAMERA_SETTINGS = {
    "framesize", "quality", "brightness", "contrast", "saturation", "sharpness",
    "whitebal", "awb_gain", "gain_ctrl", "agc_gain", "gainceiling", "exposure_ctrl",
    "aec_value", "aec2", "ae_level", "hmirror", "vflip", "wb_mode", "special_effect",
    "dcw", "bpc", "wpc", "raw_gma", "lenc", "colorbar",
}


@dataclass
class ServoPose:
    pan_deg: float
    tilt_deg: float


@dataclass
class Snapshot:
    jpeg: bytes
    utc: Optional[str]
    age_ms: Optional[int]
    servo_pan_deg: Optional[float]
    servo_tilt_deg: Optional[float]


@dataclass
class MotionEvent:
    """One PIR motion transition pushed over /ws."""
    active: bool
    servo: ServoPose
    utc: str
    seq: Optional[int] = None   # shared motion/face event counter; None on older firmware
    received_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class FaceEvent:
    """
    One face-detection pass pushed over /ws, whenever the detected face set
    changes (a face appeared/disappeared, or a recognized id changed) --
    see FaceService::consumeFrameEvent in face_service.cpp. Not sent on
    every frame the vision task processes.

    `faces` is the raw list of dicts from the device, each shaped like:
        {"id": -1, "confidence": 0.91,
         "box": {"left": 0.31, "top": 0.22, "right": 0.58, "bottom": 0.61},
         "landmarks": [{"x": 0.38, "y": 0.35}, ...]}
    `id` is -1 when the face was detected but not recognized.
    """
    faces: list
    servo: ServoPose
    utc: str
    seq: Optional[int] = None         # shared motion/face event counter; None on older firmware
    frame_seq: Optional[int] = None   # detection-pass counter, comparable with CurrentFaces.frame_seq
    received_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class CurrentFaces:
    """
    The latest processed detection pass, from GET /api/face/current. Unlike
    FaceEvent (pushed only when the face set changes), this reflects the
    boxes as they are now, give or take `age_ms`.
    """
    enabled: bool
    frame_seq: int       # 0 = no frame processed yet
    age_ms: int          # ms since capture; -1 when frame_seq == 0
    faces: list          # same dict shape as FaceEvent.faces
    servo: ServoPose
    utc: str
    received_monotonic: float = field(default_factory=time.monotonic)
    # Where the pass's time went (firmware from 2026-09-23 on, else {}):
    # frame_age_ms (frame's age when copied), wait_ms, process_ms.
    timing: dict = field(default_factory=dict)


def parse_utc(utc: str) -> Optional[float]:
    """
    Device UTC timestamp -> Unix epoch seconds (float). Accepts both the
    millisecond format ("2026-09-19T21:04:05.123Z") and the whole-second
    format older firmware sends. Returns None for "" (clock not synced yet).
    """
    if not utc:
        return None
    from datetime import datetime, timezone
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in utc else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.strptime(utc, fmt).replace(tzinfo=timezone.utc).timestamp()


def _maybe_float(value: Optional[str]) -> Optional[float]:
    return float(value) if value is not None else None


def _maybe_int(value: Optional[str]) -> Optional[int]:
    return int(value) if value is not None else None


class FaceApiClient:
    """
    Thin wrapper around a single LilyGo-Cam-RobotFace device's HTTP API.

    Example:
        face = FaceApiClient("192.168.4.1")
        face.get_status()["face_detected"]
        face.set_servo(mode="track")
        face.set_eye_target(x=0.3, y=-0.1, aperture=1.0)
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        scheme: str = "http",
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
        snapshot_port: int = 81,
    ):
        self.host = host
        self.port = port
        # Snapshots have their own server on the device, so a slow JPEG never
        # holds up status or events (port 80 redirects there too).
        self.snapshot_port = snapshot_port
        self.scheme = scheme
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"

    @property
    def snapshot_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.snapshot_port}"

    # ------------------------------------------------------------------ #
    # low-level HTTP helpers
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Optional[dict] = None, base: Optional[str] = None) -> requests.Response:
        resp = self.session.get(f"{base or self.base_url}{path}", params=params, timeout=self.timeout)
        if not resp.ok:
            raise FaceApiHTTPError(resp.status_code, resp.text)
        return resp

    def _get_json(self, path: str, params: Optional[dict] = None) -> Any:
        return self._get(path, params).json()

    def _get_ok(self, path: str, params: Optional[dict] = None) -> dict:
        """GET a command endpoint and raise if the device reports failure."""
        data = self._get_json(path, params)
        if isinstance(data, dict) and data.get("ok") is False:
            raise FaceApiError(f"device reported failure for {path}: {data}")
        return data

    # ------------------------------------------------------------------ #
    # measurements (read-only, GET)
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """
        Full JSON status: ip, camera/audio readiness, face_enabled/detected/
        id/confidence (the single primary/tracked face only -- see
        FaceEventStream for the full multi-face list with boxes/landmarks),
        eye {x,y,aperture,mode}, servo {ready,mode,pan,tilt,tracking_gain,
        tracking_rate}, motion (bool), time {synced,utc,ntp_server}.
        """
        return self._get_json("/api/status")

    def get_snapshot(self) -> Snapshot:
        """
        GET one JPEG still image, tagged with the servo pose and a
        timestamp from the moment it was captured (the device's
        X-Capture-Timestamp*/X-Servo-*-Deg response headers).
        """
        resp = self._get("/api/snapshot", base=self.snapshot_url)
        headers = resp.headers
        return Snapshot(
            jpeg=resp.content,
            utc=headers.get("X-Capture-Timestamp"),
            age_ms=_maybe_int(headers.get("X-Capture-Timestamp-Age-Ms")),
            servo_pan_deg=_maybe_float(headers.get("X-Servo-Pan-Deg")),
            servo_tilt_deg=_maybe_float(headers.get("X-Servo-Tilt-Deg")),
        )

    def get_current_faces(self) -> CurrentFaces:
        """
        GET /api/face/current: faces from the most recently processed frame
        (boxes, ids, servo pose at capture, ms UTC, age). Requires firmware
        with that endpoint (a 404 raises FaceApiHTTPError on older builds).
        """
        data = self._get_json("/api/face/current")
        return CurrentFaces(
            enabled=data["enabled"],
            frame_seq=data["frame_seq"],
            age_ms=data["age_ms"],
            faces=data.get("faces", []),
            servo=ServoPose(data["servo"]["pan"], data["servo"]["tilt"]),
            utc=data["utc"],
            timing=data.get("timing", {}),
        )

    def list_enrolled_faces(self) -> list[int]:
        """GET /api/face/list: the currently enrolled face ids."""
        return self._get_json("/api/face/list")["ids"]

    # ------------------------------------------------------------------ #
    # commands (these change device state -- GET calls, per the firmware)
    # ------------------------------------------------------------------ #

    def set_face_detection(self, enabled: bool) -> None:
        """Enables/disables face detection (GET /api/control?face=)."""
        self._get_ok("/api/control", {"face": int(enabled)})

    def set_audio_gain(self, gain: float) -> None:
        """Sets microphone digital gain, 0.0-4.0 (GET /api/control?gain=)."""
        self._get_ok("/api/control", {"gain": gain})

    def set_eye_target(self, x: float, y: float = 0.0, aperture: float = 1.0) -> None:
        """
        Sets the eye's target x/y (-1.0..1.0) and aperture (0.0..1.5).
        Only takes full, lasting effect while eye mode is "manual" -- call
        set_eye_mode("manual") first, otherwise idle scanning/face tracking
        keeps overriding this on the next control-loop tick (~5ms later).
        """
        self._get_ok("/api/control", {"x": x, "y": y, "aperture": aperture})

    def set_eye_mode(self, mode: str) -> None:
        """
        mode: "auto" (idle scanning + face tracking, the default) or
        "manual" (pauses that behavior so set_eye_target()/set_eye_target()
        sticks until you switch back to "auto").
        """
        if mode not in ("auto", "manual"):
            raise ValueError('mode must be "auto" or "manual"')
        self._get_ok("/api/eye", {"mode": mode})

    def set_face_detector(self, resize_scale: float) -> None:
        """
        Sets the scale (0.1-1.0) of the frame the detector's first stage
        searches, from the next pass (GET /api/face/detector). Higher finds
        smaller, further faces; each pass takes longer. Not persisted.
        """
        self._get_ok("/api/face/detector", {"resize_scale": resize_scale})

    def set_recognition(self, enabled: bool) -> None:
        """Enables/disables face recognition; reports the matched id in status."""
        self._get_ok("/api/face/recognize", {"enable": int(enabled)})

    def enroll_face(self, enabled: bool = True) -> None:
        """
        Arms/disarms enrollment of the next detected face. Up to 7 face IDs
        are stored in flash. Only the first face in a frame is a candidate
        when several are in view.
        """
        self._get_ok("/api/face/enroll", {"enable": int(enabled)})

    def clear_enrolled_faces(self) -> None:
        """DELETES all enrolled face IDs from flash. Cannot be undone."""
        self._get_ok("/api/face/clear", {"confirm": 1})

    def delete_enrolled_face(self, face_id: int) -> int:
        """
        DELETES one enrolled face ID from flash. Cannot be undone. Returns
        the number of IDs remaining; raises FaceApiError if the id isn't
        enrolled.
        """
        return self._get_ok("/api/face/delete", {"id": face_id})["enrolled_faces"]

    def set_servo(
        self,
        mode: Optional[str] = None,
        pan_deg: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        tracking_gain: Optional[float] = None,
        tracking_rate: Optional[float] = None,
        tilt_tracking_gain: Optional[float] = None,
    ) -> None:
        """
        MOVES THE SERVOS (if pan_deg/tilt_deg given, or mode="track" and a
        face is in view). Any subset of the parameters may be set in one
        call, mirroring GET /api/servo's flexibility:

        mode: "track" (face tracking) or "manual" (hold pan/tilt as set).
        pan_deg/tilt_deg: 0-180 degrees; only settable directly while
            mode is "manual" (face tracking drives them otherwise).
        tracking_gain: degrees of pan per normalized face offset (and of
            tilt, scaled by the frame's aspect ratio, unless tilt has its own).
        tracking_rate: smoothing speed in degrees/second.
        tilt_tracking_gain: tilt's own gain; 0 derives it from tracking_gain again.
        """
        if mode is not None and mode not in ("track", "manual"):
            raise ValueError('mode must be "track" or "manual"')
        params = {}
        if mode is not None:
            params["mode"] = mode
        if pan_deg is not None:
            params["pan"] = pan_deg
        if tilt_deg is not None:
            params["tilt"] = tilt_deg
        if tracking_gain is not None:
            params["gain"] = tracking_gain
        if tracking_rate is not None:
            params["rate"] = tracking_rate
        if tilt_tracking_gain is not None:
            params["tilt_gain"] = tilt_tracking_gain
        if not params:
            raise ValueError("set_servo() needs at least one parameter")
        self._get_ok("/api/servo", params)

    def blink_eye_closed(self) -> None:
        """Closes the aperture for 2s -- visual confirmation an HTTP command reached the device."""
        self._get_ok("/api/servo", {"eye": "closed"})

    def set_audio_destination(self, host: str, port: int) -> None:
        """
        Points the device's low-latency UDP PCM audio stream at `host:port`.
        Run tools/audio_receiver.py in ~/LilyGo-Cam-RobotFace on that host
        to actually receive/play it.
        """
        self._get_ok("/api/audio", {"host": host, "port": port})

    def probe_camera(self, frames: int = 10) -> dict:
        """
        GET /api/camera/probe: grabs `frames` (2-20) back to back. Returns
        first_age_ms (how stale the first frame handed out was), interval_ms
        (between captures) and wait_ms (per grab). Detection pauses meanwhile.
        """
        return self._get_json("/api/camera/probe", {"frames": frames})

    def set_camera(self, **settings: int) -> None:
        """
        Configures one or more camera settings in a single call, e.g.
        set_camera(brightness=1, vflip=1). Keys must be from
        CAMERA_SETTINGS (framesize, quality, brightness, contrast,
        saturation, sharpness, whitebal, awb_gain, gain_ctrl, agc_gain,
        gainceiling, exposure_ctrl, aec_value, aec2, ae_level, hmirror,
        vflip, wb_mode, special_effect, dcw, bpc, wpc, raw_gma, lenc,
        colorbar); all values are integers.
        """
        unknown = set(settings) - CAMERA_SETTINGS
        if unknown:
            raise ValueError(f"unknown camera setting(s): {sorted(unknown)}")
        if not settings:
            raise ValueError("set_camera() needs at least one setting")
        self._get_ok("/api/camera", settings)

    def set_wifi_credentials(self, ssid: str, password: str) -> None:
        """
        REBOOTS THE DEVICE. Stores WiFi credentials in flash and restarts
        to join that network. The device is unreachable at its current
        address until it reconnects (or falls back to AP mode on failure).
        """
        self._get_ok("/api/wifi", {"ssid": ssid, "password": password})

    def set_ntp_server(self, server: str) -> None:
        """Stores the NTP server in flash and resyncs; status reports UTC time."""
        self._get_ok("/api/time", {"ntp_server": server})


# ---------------------------------------------------------------------- #
# WebSocket event subscription
# ---------------------------------------------------------------------- #


class FaceEventStream:
    """
    Background-thread subscriber for the device's /ws endpoint (motion +
    face-detection events). Mirrors the background-thread pattern used by
    TeleopSession in ../vacuum-api/valetudo_client.py.

    Either register callbacks:
        events = FaceEventStream(face)
        events.on_face = lambda evt: print(evt)
        events.on_motion = lambda evt: print(evt)
        events.start()

    or just poll the cached latest event (updated on the background thread):
        events.start()
        time.sleep(1)
        print(events.last_face, events.last_motion)

    This cache is the closest equivalent to a "get current face position/
    size/id, timestamped" call -- the device doesn't expose one directly;
    see README.md.
    """

    def __init__(self, client: FaceApiClient, auto_reconnect: bool = True,
                 reconnect_delay: float = 2.0):
        if websocket is None:
            raise RuntimeError(
                "FaceEventStream requires websocket-client: pip install websocket-client")
        self.client = client
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.on_face: Optional[Callable[[FaceEvent], None]] = None
        self.on_motion: Optional[Callable[[MotionEvent], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

        self._lock = threading.Lock()
        self._last_face: Optional[FaceEvent] = None
        self._last_motion: Optional[MotionEvent] = None
        self._ws: Optional["websocket.WebSocketApp"] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def last_face(self) -> Optional[FaceEvent]:
        with self._lock:
            return self._last_face

    @property
    def last_motion(self) -> Optional[MotionEvent]:
        with self._lock:
            return self._last_motion

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            self._ws.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(self.client.ws_url, on_message=self._on_message)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:  # noqa: BLE001 - surfaced via on_error, loop must keep going
                if self.on_error:
                    self.on_error(exc)
            if not self._running or not self.auto_reconnect:
                break
            time.sleep(self.reconnect_delay)

    def _on_message(self, _ws, message: str) -> None:
        try:
            data = json.loads(message)
        except ValueError:
            return
        event = data.get("event")
        try:
            if event == "motion":
                self._handle_motion(data)
            elif event == "face":
                self._handle_face(data)
        except (KeyError, TypeError) as exc:
            if self.on_error:
                self.on_error(exc)

    def _handle_motion(self, data: dict) -> None:
        motion_event = MotionEvent(
            active=data["active"],
            servo=ServoPose(data["servo"]["pan"], data["servo"]["tilt"]),
            utc=data["utc"],
            seq=data.get("seq"),
        )
        with self._lock:
            self._last_motion = motion_event
        if self.on_motion:
            self.on_motion(motion_event)

    def _handle_face(self, data: dict) -> None:
        face_event = FaceEvent(
            faces=data.get("faces", []),
            servo=ServoPose(data["servo"]["pan"], data["servo"]["tilt"]),
            utc=data["utc"],
            seq=data.get("seq"),
            frame_seq=data.get("frame_seq"),
        )
        with self._lock:
            self._last_face = face_event
        if self.on_face:
            self.on_face(face_event)
