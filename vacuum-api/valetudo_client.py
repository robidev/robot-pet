"""
Python client for the Valetudo HTTP API (robot vacuum control).

Built against the actual Valetudo REST API (see ~/Valetudo backend source /
openapi docs), not a generic spec. A few notes on where the original
low-level-glue-logic wishlist doesn't match what Valetudo actually exposes:

- There is no bitmap/PNG map endpoint. `/api/v2/robot/state/map` returns a
  JSON description of the map (floor/wall/segment pixel layers, RLE
  compressed, plus point/line/polygon entities like the robot position and
  charger location). `get_map_image()` rasterizes that JSON into a bitmap
  locally using Pillow, since Valetudo itself only renders it client-side
  in the web UI canvas.
- There's no single "drive" command with
  {"omega", "velocity", "duration", "seqnum"}. Valetudo's manual-control
  capability (this robot has HighResolutionManualControlCapability) takes a
  one-shot {"velocity": -1..1, "angle": -180..180} vector while manual
  control is "enabled" (like a dead-man's switch: you're expected to keep
  sending vectors, or the robot stops). `drive()` below emulates the
  requested (omega, velocity, duration, seqnum) interface on top of that by
  repeatedly resending the vector for `duration` ms and then disabling
  manual control. `omega` is passed straight through as `angle` (degrees,
  -180..180) -- it is NOT an angular velocity in rad/s. Treat this as an
  approximation to tune against the real robot, not a faithful cmd_vel port.
- Some robots only support ManualControlCapability (discrete
  forward/backward/rotate_cw/rotate_ccw bumps) instead of the
  high-resolution vector control. `drive_discrete()` covers that case.

Every method that moves the robot, starts/stops cleaning, or otherwise
changes its state is a PUT/command call and is clearly marked as such in
its docstring. Nothing in this module is called automatically on import.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


class ValetudoError(Exception):
    """Base error for anything going wrong talking to Valetudo."""


class ValetudoHTTPError(ValetudoError):
    """The robot responded with a non-2xx status code."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class CapabilityNotSupportedError(ValetudoError):
    """The robot doesn't advertise the capability this call needs."""


# --- Map layer / entity type constants (from Valetudo's entity definitions) ---

LAYER_TYPE_FLOOR = "floor"
LAYER_TYPE_WALL = "wall"
LAYER_TYPE_SEGMENT = "segment"

ENTITY_TYPE_CHARGER_LOCATION = "charger_location"
ENTITY_TYPE_ROBOT_POSITION = "robot_position"
ENTITY_TYPE_GO_TO_TARGET = "go_to_target"
ENTITY_TYPE_PATH = "path"

STATUS_ERROR = "error"
STATUS_DOCKED = "docked"
STATUS_IDLE = "idle"
STATUS_RETURNING = "returning"
STATUS_CLEANING = "cleaning"
STATUS_PAUSED = "paused"
STATUS_MANUAL_CONTROL = "manual_control"
STATUS_MOVING = "moving"


@dataclass
class Pose:
    x: float
    y: float
    angle: Optional[float] = None  # degrees, 0 = North, per Valetudo convention


class ValetudoClient:
    """
    Thin wrapper around a single Valetudo instance's HTTP API.

    Example:
        vac = ValetudoClient("192.168.101.43")
        vac.get_status()          # {"value": "docked", "flag": "none", "error": None}
        vac.get_battery()         # {"level": 100, "flag": "charged"}
        vac.get_position()        # Pose(x=2560, y=2549, angle=342)
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        scheme: str = "http",
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.host = host
        self.port = port
        self.scheme = scheme
        self.timeout = timeout
        self.session = session or requests.Session()
        self._capabilities_cache: Optional[list[str]] = None

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}/api/v2"

    # ------------------------------------------------------------------ #
    # low-level HTTP helpers
    # ------------------------------------------------------------------ #

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, timeout=self.timeout)
        if not resp.ok:
            raise ValetudoHTTPError(resp.status_code, resp.text)
        if not resp.content:
            return None
        return resp.json()

    def _put(self, path: str, body: dict) -> None:
        url = f"{self.base_url}{path}"
        resp = self.session.put(url, json=body, timeout=self.timeout)
        if not resp.ok:
            raise ValetudoHTTPError(resp.status_code, resp.text)

    # ------------------------------------------------------------------ #
    # capability discovery
    # ------------------------------------------------------------------ #

    def get_capabilities(self, refresh: bool = False) -> list[str]:
        """GET the list of capability type names this robot supports."""
        if refresh or self._capabilities_cache is None:
            self._capabilities_cache = self._get("/robot/capabilities")
        return self._capabilities_cache

    def has_capability(self, capability_type: str) -> bool:
        return capability_type in self.get_capabilities()

    def _require_capability(self, capability_type: str) -> None:
        if not self.has_capability(capability_type):
            raise CapabilityNotSupportedError(
                f"Robot does not advertise '{capability_type}'. "
                f"Available: {self.get_capabilities()}"
            )

    # ------------------------------------------------------------------ #
    # measurements (read-only, GET)
    # ------------------------------------------------------------------ #

    def get_robot_info(self) -> dict:
        """manufacturer / modelName / modelDetails / implementation."""
        return self._get("/robot")

    def get_properties(self) -> dict:
        """Robot-specific flat key/value properties (e.g. firmware version)."""
        return self._get("/robot/properties")

    def get_state(self) -> dict:
        """Full polled state: {"attributes": [...], "map": {...}}."""
        return self._get("/robot/state")

    def get_state_attributes(self) -> list[dict]:
        """Just the attributes array (status/battery/fan speed/etc.)."""
        return self._get("/robot/state/attributes")

    def get_map(self) -> dict:
        """
        Raw ValetudoMap JSON: size, pixelSize, layers (floor/wall/segment,
        RLE-compressed pixels), and entities (robot_position,
        charger_location, path, go_to_target, ...).
        """
        return self._get("/robot/state/map")

    def get_map_image(self, scale: int = 8, crop_to_content: bool = True, padding: int = 15):
        """
        Rasterize the current map into a Pillow Image (bitmap), since
        Valetudo doesn't expose one directly. Requires `pillow`
        (pip install pillow).

        Colors: floor = light grey, walls = dark grey, segments = a
        rotating palette, robot = blue dot, charger = green dot.

        By default the image is cropped to the bounding box of the actual
        floor/wall/segment layers (+ `padding` map units), then scaled up
        by `scale`. Without this, the room is a tiny speck in the middle
        of Valetudo's full (e.g. 5120x5120) coordinate space. Pass
        `crop_to_content=False` to get the raw full-canvas image instead.

        Note: the bounding box intentionally ignores the `path` entity
        (the robot's cleaning trail), which can extend far outside the
        mapped room and would otherwise blow the crop back up to
        near-full-canvas size.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError(
                "get_map_image() requires pillow: pip install pillow"
            ) from exc

        map_data = self.get_map()
        size = map_data["size"]
        img = Image.new("RGB", (size["x"], size["y"]), (20, 20, 20))
        draw = ImageDraw.Draw(img)

        segment_palette = [
            (255, 179, 0), (0, 191, 255), (255, 99, 71),
            (154, 205, 50), (218, 112, 214), (255, 215, 0),
        ]

        content_layers = (LAYER_TYPE_FLOOR, LAYER_TYPE_WALL, LAYER_TYPE_SEGMENT)
        xs_min, xs_max, ys_min, ys_max = [], [], [], []

        for layer in map_data.get("layers", []):
            pixels = _decompress_pixels(layer.get("compressedPixels") or layer.get("pixels") or [])
            if layer["type"] == LAYER_TYPE_FLOOR:
                color = (200, 200, 200)
            elif layer["type"] == LAYER_TYPE_WALL:
                color = (60, 60, 60)
            else:  # segment
                seg_id = layer.get("metaData", {}).get("segmentId", "0")
                color = segment_palette[hash(seg_id) % len(segment_palette)]

            for x, y in _pairwise(pixels):
                draw.point((x, y), fill=color)

            if layer["type"] in content_layers:
                dims = layer.get("dimensions", {})
                if dims:
                    xs_min.append(dims["x"]["min"])
                    xs_max.append(dims["x"]["max"])
                    ys_min.append(dims["y"]["min"])
                    ys_max.append(dims["y"]["max"])

        for entity in map_data.get("entities", []):
            if entity["type"] == ENTITY_TYPE_ROBOT_POSITION:
                x, y = entity["points"]
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 120, 255))
            elif entity["type"] == ENTITY_TYPE_CHARGER_LOCATION:
                x, y = entity["points"]
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 200, 0))

        if crop_to_content and xs_min:
            x0 = max(0, min(xs_min) - padding)
            y0 = max(0, min(ys_min) - padding)
            x1 = min(size["x"], max(xs_max) + padding)
            y1 = min(size["y"], max(ys_max) + padding)
            img = img.crop((x0, y0, x1, y1))

        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        return img

    def _find_point_entity(self, entity_type: str) -> Optional[dict]:
        for entity in self.get_map().get("entities", []):
            if entity["type"] == entity_type:
                return entity
        return None

    def get_position(self) -> Optional[Pose]:
        """Robot's current (x, y, angle) on the map, or None if unavailable."""
        entity = self._find_point_entity(ENTITY_TYPE_ROBOT_POSITION)
        if entity is None:
            return None
        x, y = entity["points"]
        angle = entity.get("metaData", {}).get("angle")
        return Pose(x=x, y=y, angle=angle)

    def get_angle(self) -> Optional[float]:
        """Robot's heading in degrees (0 = North), or None if unavailable."""
        pose = self.get_position()
        return pose.angle if pose else None

    def get_charger_position(self) -> Optional[Pose]:
        entity = self._find_point_entity(ENTITY_TYPE_CHARGER_LOCATION)
        if entity is None:
            return None
        x, y = entity["points"]
        return Pose(x=x, y=y)

    def _find_attribute(self, class_name: str) -> Optional[dict]:
        for attr in self.get_state_attributes():
            if attr.get("__class") == class_name or attr.get("type") == class_name:
                return attr
        return None

    def get_battery(self) -> dict:
        """{"level": 0-100, "flag": "none"|"charging"|"discharging"|"charged"}."""
        attr = self._find_attribute("BatteryStateAttribute") or {}
        return {"level": attr.get("level"), "flag": attr.get("flag")}

    def get_status(self) -> dict:
        """
        {"value": "docked"|"idle"|"returning"|"cleaning"|"paused"|
                   "manual_control"|"moving"|"error",
         "flag": "none"|"zone"|"segment"|"spot"|"target"|"resumable"|"mapping",
         "error": {...} or None}
        """
        attr = self._find_attribute("StatusStateAttribute") or {}
        return {
            "value": attr.get("value"),
            "flag": attr.get("flag"),
            "error": attr.get("error"),
        }

    def is_docked(self) -> bool:
        return self.get_status().get("value") == STATUS_DOCKED

    def is_cleaning(self) -> bool:
        return self.get_status().get("value") == STATUS_CLEANING

    def is_paused(self) -> bool:
        return self.get_status().get("value") == STATUS_PAUSED

    def is_idle(self) -> bool:
        return self.get_status().get("value") == STATUS_IDLE

    def is_moving(self) -> bool:
        return self.get_status().get("value") in (STATUS_MOVING, STATUS_MANUAL_CONTROL)

    def has_error(self) -> bool:
        return self.get_status().get("value") == STATUS_ERROR

    # ------------------------------------------------------------------ #
    # commands (these move the robot / change its state -- PUT calls)
    # ------------------------------------------------------------------ #

    def start_cleaning(self) -> None:
        """MOVES THE ROBOT. Starts/resumes cleaning ("play")."""
        self._require_capability("BasicControlCapability")
        self._put("/robot/capabilities/BasicControlCapability", {"action": "start"})

    def pause_cleaning(self) -> None:
        """MOVES THE ROBOT (stops in place). Pauses the current job."""
        self._require_capability("BasicControlCapability")
        self._put("/robot/capabilities/BasicControlCapability", {"action": "pause"})

    def stop_cleaning(self) -> None:
        """MOVES THE ROBOT (stops in place). Stops the current job entirely."""
        self._require_capability("BasicControlCapability")
        self._put("/robot/capabilities/BasicControlCapability", {"action": "stop"})

    def go_to_dock(self) -> None:
        """MOVES THE ROBOT. Sends it home to the charging dock."""
        self._require_capability("BasicControlCapability")
        self._put("/robot/capabilities/BasicControlCapability", {"action": "home"})

    def go_to(self, x: float, y: float) -> None:
        """MOVES THE ROBOT. Drives to the given map coordinate (x, y)."""
        self._require_capability("GoToLocationCapability")
        self._put(
            "/robot/capabilities/GoToLocationCapability",
            {"action": "goto", "coordinates": {"x": x, "y": y}},
        )

    def locate(self) -> None:
        """Makes the robot beep/announce its location. Doesn't move it."""
        self._require_capability("LocateCapability")
        self._put("/robot/capabilities/LocateCapability", {"action": "locate"})

    # -- manual / teleop control -------------------------------------- #

    def _manual_capability_path(self) -> str:
        if self.has_capability("HighResolutionManualControlCapability"):
            return "/robot/capabilities/HighResolutionManualControlCapability"
        self._require_capability("ManualControlCapability")
        return "/robot/capabilities/ManualControlCapability"

    def manual_control_active(self) -> bool:
        return bool(self._get(self._manual_capability_path()).get("enabled"))

    def enable_manual_control(self) -> None:
        """MOVES THE ROBOT (arms teleop mode). Required before drive_vector()."""
        self._put(self._manual_capability_path(), {"action": "enable"})

    def disable_manual_control(self) -> None:
        """Leaves teleop mode."""
        self._put(self._manual_capability_path(), {"action": "disable"})

    def drive_vector(self, velocity: float, angle: float) -> None:
        """
        MOVES THE ROBOT. One-shot manual-control move.

        velocity: -1..1 (forward/backward)
        angle:    -180..180 degrees, steering offset

        Requires HighResolutionManualControlCapability and manual control to
        already be enabled (see enable_manual_control()). Valetudo expects
        this to be resent periodically -- a single call will typically only
        move the robot for a short burst before it stops on its own.
        """
        self._require_capability("HighResolutionManualControlCapability")
        if not -1 <= velocity <= 1:
            raise ValueError("velocity must be within -1..1")
        if not -180 <= angle <= 180:
            raise ValueError("angle must be within -180..180 degrees")
        self._put(
            "/robot/capabilities/HighResolutionManualControlCapability",
            {"action": "move", "vector": {"velocity": velocity, "angle": angle}},
        )

    def drive_discrete(self, direction: str) -> None:
        """
        MOVES THE ROBOT. Discrete manual-control bump, for robots that only
        support ManualControlCapability (no HighResolutionManualControlCapability).

        direction: "forward" | "backward" | "rotate_clockwise" | "rotate_counterclockwise"
        """
        self._require_capability("ManualControlCapability")
        valid = {"forward", "backward", "rotate_clockwise", "rotate_counterclockwise"}
        if direction not in valid:
            raise ValueError(f"direction must be one of {valid}")
        self._put(
            "/robot/capabilities/ManualControlCapability",
            {"action": "move", "movementCommand": direction},
        )

    def drive(
        self,
        omega: float,
        velocity: float,
        duration: int,
        seqnum: Optional[int] = None,
        update_interval: float = 0.15,
    ) -> None:
        """
        MOVES THE ROBOT for approximately `duration` milliseconds.

        This emulates the requested {"omega", "velocity", "duration",
        "seqnum"} drive interface on top of Valetudo's manual-control
        vector API, which only knows a one-shot (velocity, angle) command
        while manual control is enabled. There is no true continuous
        angular-velocity control here:

          - `omega` is passed straight through as the steering `angle`
            (-180..180 degrees), NOT an angular velocity in rad/s.
          - `velocity` is Valetudo's -1..1 forward/backward speed.
          - the vector is resent every `update_interval` seconds for the
            duration (Valetudo needs repeated commands or it stops), then
            manual control is disabled and the robot halts.
          - `seqnum` is accepted for API-compatibility with the
            (omega/velocity/duration/seqnum) drive spec but is not sent to
            Valetudo; it's just yours to use for your own bookkeeping/logging.

        This call blocks for `duration` milliseconds. Enables manual
        control if not already enabled, and disables it again at the end.
        """
        self._require_capability("HighResolutionManualControlCapability")
        angle = max(-180.0, min(180.0, omega))
        velocity = max(-1.0, min(1.0, velocity))

        was_active = self.manual_control_active()
        if not was_active:
            self.enable_manual_control()
        try:
            deadline = time.monotonic() + duration / 1000.0
            while time.monotonic() < deadline:
                self.drive_vector(velocity, angle)
                time.sleep(update_interval)
        finally:
            if not was_active:
                self.disable_manual_control()

    def stop_drive(self) -> None:
        """Immediately halts teleop movement by disabling manual control."""
        self.disable_manual_control()


class TeleopSession:
    """
    Optional helper for sustained manual driving from something like a
    joystick/UI loop: keeps resending the last commanded vector on a
    background thread so Valetudo's manual-control dead-man's switch
    doesn't time out, until you call stop() or change the vector.

    Example:
        vac = ValetudoClient("192.168.101.43")
        teleop = TeleopSession(vac)
        teleop.start()
        teleop.set_vector(velocity=0.3, angle=-30)   # MOVES THE ROBOT
        ...
        teleop.stop()
    """

    def __init__(self, client: ValetudoClient, update_interval: float = 0.15):
        self.client = client
        self.update_interval = update_interval
        self._velocity = 0.0
        self._angle = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """MOVES THE ROBOT (arms teleop mode) and starts the resend loop."""
        if self._running:
            return
        self.client.enable_manual_control()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_vector(self, velocity: float, angle: float) -> None:
        """Updates the commanded (velocity, angle); takes effect on next resend."""
        with self._lock:
            self._velocity = velocity
            self._angle = angle

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                velocity, angle = self._velocity, self._angle
            try:
                self.client.drive_vector(velocity, angle)
            except ValetudoError:
                pass
            time.sleep(self.update_interval)

    def stop(self) -> None:
        """Stops the resend loop and disables manual control."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.client.disable_manual_control()


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #


def _decompress_pixels(compressed: list[int]) -> list[int]:
    """RLE-decode Valetudo's [xStart, y, count, ...] compressed pixel triplets."""
    out: list[int] = []
    for i in range(0, len(compressed), 3):
        x_start, y, count = compressed[i], compressed[i + 1], compressed[i + 2]
        for j in range(count):
            out.append(x_start + j)
            out.append(y)
    return out


def _pairwise(flat: list[int]):
    for i in range(0, len(flat), 2):
        yield flat[i], flat[i + 1]
