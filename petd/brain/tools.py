"""
The pet's tool registry: one definition per ability, used by every LLM
backend.

- The Claude CLI reaches these over MCP: it spawns petd/mcp_shim/robot_mcp.py,
  which forwards to petd's local HTTP API (GET /tools, POST /tool/{name}).
  The CLI runs MCP servers as its own children, so they can't share memory
  with petd; the HTTP hop keeps one source of truth.
- Other backends (ollama) call registry.call() directly.

Tools that take time (driving, searching) start a *behavior* and return
immediately; the outcome arrives later as an event (see PLAN.md 4.5/4.6).
Anything destructive or unrelated to being a pet (clearing faces, wifi,
cleaning) is deliberately absent, not just disallowed.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

Handler = Callable[[dict], Awaitable[Any]]


class ToolError(Exception):
    """Reported back to the model as a failed tool result, not a crash."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict            # JSON Schema for the arguments
    handler: Handler
    # A tool result is either plain data (JSON-encoded for the model) or
    # {"image": bytes, "mime": "image/jpeg"} for vision.
    returns_image: bool = False


@dataclass
class ToolResult:
    text: Optional[str] = None
    image_b64: Optional[str] = None
    mime: str = "image/jpeg"
    is_error: bool = False
    meta: dict = field(default_factory=dict)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, schema: Optional[dict] = None,
             returns_image: bool = False):
        def decorate(fn: Handler) -> Handler:
            self.add(Tool(name, description,
                          schema or {"type": "object", "properties": {}},
                          fn, returns_image))
            return fn
        return decorate

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "schema": t.schema}
                for t in self._tools.values()]

    async def call(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(text=f"no such tool: {name}", is_error=True)
        started = time.monotonic()
        try:
            value = await tool.handler(arguments or {})
        except ToolError as exc:
            log.info("tool %s refused: %s", name, exc)
            return ToolResult(text=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - the model gets to hear about it
            log.exception("tool %s failed", name)
            return ToolResult(text=f"{type(exc).__name__}: {exc}", is_error=True)
        log.info("tool %s(%s) -> %.0fms", name, arguments or {}, (time.monotonic() - started) * 1000)
        if isinstance(value, dict) and "image" in value:
            return ToolResult(text=value.get("text"),
                              image_b64=base64.b64encode(value["image"]).decode(),
                              mime=value.get("mime", "image/jpeg"))
        return ToolResult(text=value if isinstance(value, str) else _as_json(value))


def _as_json(value: Any) -> str:
    import json

    from ..api.server import to_jsonable
    return json.dumps(to_jsonable(value), default=str)


def build_registry(pet: "App") -> ToolRegistry:
    """The MVP tool set (PLAN.md 4.5). Later clusters add movement and memory."""
    registry = ToolRegistry()

    def _face_name(face) -> Optional[str]:
        if not face.recognized:
            return None
        person = pet.db.person_by_slot(face.id) if pet.db is not None else None
        return person.name if person else f"a face I stored without a name (#{face.id})"

    def require_face():
        if pet.face is None:
            raise ToolError("I have no head attached right now")
        return pet.face

    @registry.tool(
        "get_senses",
        "What I can sense right now: battery, what I'm doing, where I am, who is in view, "
        "and whether I'm speaking. Cheap - call it whenever you need current state.")
    async def get_senses(args: dict) -> dict:
        out: dict = {"time": time.strftime("%H:%M"), "pet_name": pet.cfg.pet.name}
        if pet.vacuum:
            s = pet.vacuum.state
            out["body"] = {
                "reachable": s.reachable, "doing": s.status, "battery_percent": s.battery_level,
                "charging": s.battery_flag, "docked": s.docked,
                "position": None if not s.pose else {"x": s.pose.x, "y": s.pose.y, "heading": s.pose.angle},
            }
        if pet.face:
            f = pet.face.state
            frame = await pet.face.current_faces()
            out["head"] = {"reachable": f.reachable, "pan_deg": f.pan_deg, "tilt_deg": f.tilt_deg,
                           "servo_mode": f.servo_mode, "known_faces_stored": f.enrolled}
            out["sees"] = [
                {"who": _face_name(face),
                 "where": _describe_position(face.cx),
                 "size": round(face.h, 2), "confidence": round(face.confidence, 2)}
                for face in frame.faces
            ]
            out["someone_present"] = pet.face.presence.present
            if pet.people is not None:
                names, strangers = pet.people.who_is_here()
                # Recognition flickers frame to frame; this is who was
                # recognized at any point since they came into view.
                out["people_here"] = {"known": names, "unrecognized": strangers}
        if pet.speaker:
            out["speaking"] = pet.speaker.speaking
        return out

    @registry.tool(
        "look",
        "Take a photo with my camera and look at it. Use it to answer questions about what I can "
        "see, or when curious about my surroundings.",
        returns_image=True)
    async def look(args: dict) -> dict:
        snap = await require_face().snapshot()
        if snap is None:
            raise ToolError("my camera returned nothing")
        return {"image": snap.jpeg, "mime": "image/jpeg",
                "text": f"photo taken at pan {snap.servo_pan_deg}, tilt {snap.servo_tilt_deg}"}

    @registry.tool(
        "look_direction",
        "Point my head. pan: 0 is far right, 90 straight ahead, 180 far left. "
        "tilt: 90 is level, higher looks up. Turns off face tracking while I hold the pose.",
        {"type": "object",
         "properties": {"pan": {"type": "number", "minimum": 0, "maximum": 180},
                        "tilt": {"type": "number", "minimum": 0, "maximum": 180}},
         "required": []})
    async def look_direction(args: dict) -> str:
        face = require_face()
        pan, tilt = args.get("pan"), args.get("tilt")
        if pan is None and tilt is None:
            raise ToolError("give me a pan and/or tilt angle")
        await face.set_servo(mode="manual")
        await face.set_servo(pan_deg=_clamp(pan, 0, 180), tilt_deg=_clamp(tilt, 0, 180))
        return "looking there"

    @registry.tool(
        "track_faces",
        "Turn face tracking on or off. On means my head follows whoever I'm looking at, "
        "keeping eye contact. Turn it on when someone is with me. It only moves my head: "
        "it does not learn or remember anyone's face.",
        {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]})
    async def track_faces(args: dict) -> str:
        await require_face().set_servo(mode="track" if args.get("on", True) else "manual")
        return "tracking on" if args.get("on", True) else "tracking off"

    @registry.tool(
        "go_home",
        "Drive back to my charging dock. Use it when my battery is low, when I'm told to go "
        "home or to sleep, or when I've finished what I was doing.")
    async def go_home(args: dict) -> str:
        if pet.vacuum is None:
            raise ToolError("my wheels aren't connected")
        await pet.vacuum.dock()
        return "heading to the dock"

    @registry.tool(
        "stop",
        "Stop moving and stop talking immediately. Use it the moment someone tells me to stop.")
    async def stop(args: dict) -> str:
        await pet.stop_everything("tool")
        return "stopped"

    if pet.people is not None:
        _add_memory_tools(registry, pet)
    return registry


def _add_memory_tools(registry: ToolRegistry, pet: "App") -> None:
    """The people and memory tools (PLAN.md 4.5, 4.7; cluster E3)."""
    from ..memory.people import FAMILIARITY_WORDS, ago
    people, db = pet.people, pet.db
    name_arg = {"type": "string", "description": "The person's name, as they said it"}

    def require_person(name: str):
        person = db.person_by_name(name or "")
        if person is None:
            raise ToolError(f"I don't know anyone called {name}")
        return person

    @registry.tool(
        "remember_face",
        "Memorize the face of the person in front of me, under their name, so I recognize them "
        "from now on. Only when they ask me to remember them or agree to it. Exactly one person "
        "must be in view, facing me. Takes a few seconds; the result says whether it worked.",
        {"type": "object",
         "properties": {"name": name_arg,
                        "insist": {"type": "boolean", "description":
                                   "Store the face even though it looks like someone I know, or "
                                   "retake a face I already have. Only after a first attempt "
                                   "told me to."}},
         "required": ["name"]})
    async def remember_face(args: dict) -> str:
        return await people.enroll(args.get("name", ""), insist=bool(args.get("insist")))

    @registry.tool(
        "forget_person",
        "Permanently delete someone: their stored face and everything I know about them. Only "
        "when a person explicitly asks me to forget them (or someone by name). Never on my own "
        "initiative, and never on a vague 'forget it'.",
        {"type": "object", "properties": {"name": name_arg}, "required": ["name"]})
    async def forget_person(args: dict) -> str:
        return await people.forget(args.get("name", ""))

    @registry.tool(
        "who_do_i_know",
        "Everyone I know by name, whether I have their face stored, and when I last saw them.")
    async def who_do_i_know(args: dict) -> list:
        now = time.time()
        return [{"name": p.name, "nickname": p.nickname,
                 "face_stored": p.face_slot is not None,
                 "familiarity": FAMILIARITY_WORDS[min(p.familiarity, 3)],
                 "last_seen": None if p.last_seen_at is None else ago(now - p.last_seen_at) + " ago"}
                for p in db.people()]

    @registry.tool(
        "recall_person",
        "What I remember about someone: my notes, things they told me, and when and where I "
        "last saw them.",
        {"type": "object", "properties": {"name": name_arg}, "required": ["name"]})
    async def recall_person(args: dict) -> dict:
        person = require_person(args.get("name", ""))
        now = time.time()
        return {"name": person.name, "nickname": person.nickname, "notes": person.notes or None,
                "face_stored": person.face_slot is not None,
                "familiarity": FAMILIARITY_WORDS[min(person.familiarity, 3)],
                "times_met": person.interactions,
                "last_seen": (None if person.last_seen_at is None
                              else ago(now - person.last_seen_at) + " ago"),
                "in_view_now": person.id in people.present,
                "facts": [f.text for f in db.facts(about=person.id, limit=10)]}

    @registry.tool(
        "note_about_person",
        "Remember something about someone I know: a preference, a running joke, something they "
        "told me. Short, one fact per call. Can also set the nickname I use for them.",
        {"type": "object",
         "properties": {"name": name_arg,
                        "note": {"type": "string", "maxLength": 300},
                        "nickname": {"type": "string", "maxLength": 40}},
         "required": ["name"]})
    async def note_about_person(args: dict) -> str:
        person = require_person(args.get("name", ""))
        note, nickname = (args.get("note") or "").strip(), (args.get("nickname") or "").strip()
        if not note and not nickname:
            raise ToolError("give me a note or a nickname")
        if note:
            db.add_fact(note, about=person.id)
        if nickname:
            db.set_nickname(person.id, nickname)
        return f"noted about {person.name}"

    @registry.tool(
        "remember_fact",
        "Remember something general for later: about the house, the routine, or myself. Not for "
        "facts about one person (use note_about_person). Short, one fact per call.",
        {"type": "object", "properties": {"text": {"type": "string", "maxLength": 300}},
         "required": ["text"]})
    async def remember_fact(args: dict) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            raise ToolError("remember what, exactly?")
        db.add_fact(text)
        return "noted"


def _clamp(value: Optional[float], low: float, high: float) -> Optional[float]:
    return None if value is None else max(low, min(high, float(value)))


def _describe_position(cx: float) -> str:
    """
    Horizontal position in the camera frame, in words the model can use.

    NOTE: which side of the image is the robot's left depends on the camera
    mount and its hmirror setting; C3's calibration pins that down (and
    should replace this mapping with the calibrated sign).
    """
    if cx < 0.35:
        return "to my right"
    if cx > 0.65:
        return "to my left"
    return "straight ahead"
