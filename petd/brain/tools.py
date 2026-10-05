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

from ..config import CalibrationConfig
from .expressions import head_offset
from ..events import ToolRan

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
    def __init__(self, bus=None):
        self._tools: dict[str, Tool] = {}
        self.bus = bus              # each run is published as ToolRan (PLAN.md 4.9)

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
        started, started_wall = time.monotonic(), time.time()

        def ran(is_error: bool) -> None:
            if self.bus is not None:
                self.bus.publish(ToolRan(name=name, started=started_wall,
                                         duration_s=time.monotonic() - started, is_error=is_error))
        try:
            value = await tool.handler(arguments or {})
        except ToolError as exc:
            log.info("tool %s refused: %s", name, exc)
            ran(True)
            return ToolResult(text=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - the model gets to hear about it
            log.exception("tool %s failed", name)
            ran(True)
            return ToolResult(text=f"{type(exc).__name__}: {exc}", is_error=True)
        ran(False)
        log.info("tool %s(%s) -> %.0fms", name, arguments or {}, (time.monotonic() - started) * 1000)
        if isinstance(value, dict) and "image" in value:
            return ToolResult(text=value.get("text"),
                              image_b64=base64.b64encode(value["image"]).decode(),
                              mime=value.get("mime", "image/jpeg"))
        return ToolResult(text=value if isinstance(value, str) else _as_json(value))


def _as_json(value: Any) -> str:
    import json

    from ..jsonable import to_jsonable
    return json.dumps(to_jsonable(value), default=str)


def build_registry(pet: "App") -> ToolRegistry:
    """The MVP tool set (PLAN.md 4.5). Later clusters add movement and memory."""
    registry = ToolRegistry(pet.bus)

    def _face_name(face) -> Optional[str]:
        if not face.recognized:
            return None
        person = pet.db.person_by_slot(face.id) if pet.db is not None else None
        return person.name if person else f"a face I stored without a name (#{face.id})"

    def require_face():
        if pet.face is None:
            raise ToolError("I have no head attached right now")
        if not pet.face.state.reachable:
            # Each call to it would take seconds to fail.
            raise ToolError("my head is offline right now: no camera, and I can't move it")
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
        frame = None
        if pet.face and pet.face.state.reachable:
            # The rest of the senses don't depend on the head: if it has gone
            # since its last status poll, say so rather than fail them all.
            try:
                frame = await pet.face.current_faces()
            except Exception as exc:  # noqa: BLE001
                log.info("get_senses: the head didn't answer: %s", exc)
        if pet.face and frame is None:
            out["head"] = {"reachable": False}
            out["sees"] = "nothing: my head is offline, so no camera and no face detection"
        elif pet.face:
            f = pet.face.state
            out["head"] = {"reachable": f.reachable, "pan_deg": f.pan_deg, "tilt_deg": f.tilt_deg,
                           "servo_mode": f.servo_mode, "known_faces_stored": f.enrolled}
            out["sees"] = [
                {"who": _face_name(face),
                 "where": _describe_position(face.cx, pet.cfg.calibration),
                 "size": round(face.h, 2), "confidence": round(face.confidence, 2)}
                for face in frame.faces
            ]
            out["someone_present"] = pet.face.presence.present
            out["face_recognition"] = pet.recognition_on
            if pet.people is not None and pet.recognition_on:
                names, strangers = pet.people.who_is_here()
                # Recognition flickers frame to frame; this is who was
                # recognized at any point since they came into view.
                out["people_here"] = {"known": names, "unrecognized": strangers}
        if pet.speaker:
            out["speaking"] = pet.speaker.speaking
        if pet.db is not None:
            out["places_i_know"] = pet.db.places()
        if pet.motion is not None:
            out["on_the_move"] = pet.moving
        return out

    @registry.tool(
        "look",
        "Take a photo with my camera and look at it. Use it to answer questions about what I can "
        "see, or when curious about my surroundings.",
        returns_image=True)
    async def look(args: dict) -> dict:
        snap = await require_face().snapshot()
        if pet.brain is not None and pet.brain.expressions is not None:
            pet.brain.expressions.extend_hold()     # looking at the held view: keep holding
        if snap is None:
            raise ToolError("my camera is not working right now: no photo came back, so I saw nothing at all")
        return {"image": snap.jpeg, "mime": "image/jpeg",
                "text": f"photo taken at pan {snap.servo_pan_deg}, tilt {snap.servo_tilt_deg}"}

    @registry.tool(
        "look_direction",
        _look_direction_help(pet.cfg.calibration, pet.cfg.face.tilt_min_deg, pet.cfg.face.tilt_max_deg,
                             pet.cfg.face.look_turn_deg, pet.cfg.face.look_tilt_deg, pet.cfg.face.look_hold_s),
        {"type": "object",
         "properties": {"direction": {"type": "string", "enum": list(LOOK_DIRECTIONS) + ["ahead"]},
                        "pan": {"type": "number", "minimum": 0, "maximum": 180},
                        "tilt": {"type": "number", "minimum": pet.cfg.face.tilt_min_deg,
                                 "maximum": pet.cfg.face.tilt_max_deg}},
         "required": []})
    async def look_direction(args: dict) -> str:
        face = require_face()
        fcfg, cal = pet.cfg.face, pet.cfg.calibration
        direction, pan, tilt = args.get("direction"), args.get("pan"), args.get("tilt")
        if direction is None and pan is None and tilt is None:
            raise ToolError("give me a direction, or a pan and/or tilt angle")
        if direction is not None and direction != "ahead" and direction not in LOOK_DIRECTIONS:
            raise ToolError(f"direction is one of {', '.join(LOOK_DIRECTIONS)} or ahead")
        expressions = pet.brain.expressions if pet.brain is not None else None
        if direction == "ahead":
            pan = cal.pan_forward_deg if pan is None else pan
            tilt = cal.tilt_level_deg if tilt is None else tilt
        elif direction is not None:
            left, up = LOOK_DIRECTIONS[direction]
            pan_offset, tilt_offset = head_offset(left * fcfg.look_turn_deg, up * fcfg.look_tilt_deg, cal)
            now_pan, now_tilt = (expressions.base_pose() if expressions is not None
                                 else (face.state.pan_deg, face.state.tilt_deg))
            if pan_offset and pan is None:
                pan = (now_pan if now_pan is not None else cal.pan_forward_deg) + pan_offset
            if tilt_offset and tilt is None:
                tilt = (now_tilt if now_tilt is not None else cal.tilt_level_deg) + tilt_offset
        pan, tilt = _clamp(pan, 0, 180), _clamp(tilt, fcfg.tilt_min_deg, fcfg.tilt_max_deg)
        if expressions is not None:
            await expressions.hold(pan, tilt, fcfg.look_hold_s)
        else:
            await face.set_servo(mode="manual")
            await face.set_servo(pan_deg=pan, tilt_deg=tilt)
        where = ", ".join(f"{name} {value:g}" for name, value in (("pan", pan), ("tilt", tilt)) if value is not None)
        return f"looking there ({where})"

    @registry.tool(
        "track_faces",
        "Turn face tracking on or off. On means my head follows whoever I'm looking at, "
        "keeping eye contact. Turn it on when someone is with me. It only moves my head: "
        "it does not learn or remember anyone's face.",
        {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]})
    async def track_faces(args: dict) -> str:
        if pet.brain is not None and pet.brain.expressions is not None:
            pet.brain.expressions.end_hold()
        await require_face().set_servo(mode="track" if args.get("on", True) else "manual")
        return "tracking on" if args.get("on", True) else "tracking off"

    @registry.tool(
        "go_home",
        "Drive back to my charging dock. Use it when my battery is low, when I'm told to go "
        "home or to sleep, or when I've finished what I was doing.")
    async def go_home(args: dict) -> str:
        if pet.vacuum is None or pet.dock is None:
            raise ToolError("my wheels aren't connected")
        if pet.vacuum.state.docked:
            return "I'm already on my dock"
        await pet.go_home()
        return "heading to the dock; I'm told when I'm on it, or if I couldn't find it"

    @registry.tool(
        "stop",
        "Stop moving and stop talking immediately. Use it the moment someone tells me to stop.")
    async def stop(args: dict) -> str:
        await pet.stop_everything("tool")
        return "stopped"

    if pet.people is not None:
        _add_memory_tools(registry, pet)
    if pet.motion is not None:
        _add_motion_tools(registry, pet)
    return registry


def _add_motion_tools(registry: ToolRegistry, pet: "App") -> None:
    """Turning, moving and named places (PLAN.md 4.5; cluster C4)."""
    import asyncio

    from ..io.vacuum import MapPose
    from ..spatial.dock import await_arrival
    motion, vacuum, cfg = pet.motion, pet.vacuum, pet.cfg.motion

    def check_can_move(leaving_dock_ok: bool = False) -> None:
        state = vacuum.state
        if not state.reachable:
            raise ToolError("my wheels aren't answering")
        if pet.moving:
            raise ToolError("I'm already on the move; that has to finish (or be stopped) first")
        if state.docked:
            if not leaving_dock_ok:
                raise ToolError("I'm on my dock: turning or backing up here would scrape my "
                                "contacts. To leave it, drive straight forward first (move with "
                                "positive cm), or go to a named place.")
            if (state.battery_level or 0) < cfg.min_battery_to_leave:
                raise ToolError(f"my battery is at {state.battery_level}%, too low to leave the "
                                "dock; I'd only have to come straight back")

    warmup = ("My lidar spins up first, so the first motion takes about ten seconds to start; "
              "say something before calling it rather than leaving a silence.")

    async def run_to_completion(coro, describe: str) -> str:
        """Turns and moves finish before the tool returns, so what comes next
        (a look, another move) happens where I ended up, not on the way."""
        task = pet.start_motion(coro, describe, report=False)
        try:
            outcome, ok = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                return f"{describe} was cut short: I was told to stop"
            raise
        if not ok:
            raise ToolError(outcome)
        return outcome

    @registry.tool(
        "turn",
        "Turn my whole body in place. Positive degrees turn left (counter-clockwise), negative "
        "turn right; up to 180 either way. Returns when the turn is done. " + warmup,
        {"type": "object",
         "properties": {"degrees": {"type": "number", "minimum": -180, "maximum": 180}},
         "required": ["degrees"]})
    async def turn(args: dict) -> str:
        degrees = float(args.get("degrees") or 0)
        if abs(degrees) < 3:
            raise ToolError("that's not a turn, that's a twitch")
        check_can_move()

        async def run():
            result = await motion.turn_by(degrees)
            return result.describe(), result.ok
        return await run_to_completion(run(), f"turn {degrees:+.0f} degrees")

    @registry.tool(
        "move",
        "Drive straight: positive centimetres forward, negative backward, up to 100. Slow (about "
        "12 cm/s) and only my bumpers see obstacles, so only when the way looks clear. Returns "
        "when I've stopped. Driving forward is also how I leave my dock. " + warmup,
        {"type": "object",
         "properties": {"cm": {"type": "number", "minimum": -100, "maximum": 100}},
         "required": ["cm"]})
    async def move(args: dict) -> str:
        cm = float(args.get("cm") or 0)
        if abs(cm) < 3:
            raise ToolError("too small to bother the wheels with")
        # Straight forward is how the robot leaves its dock (driven by hand
        # it came off cleanly); turning or reversing on it would scrape the
        # contacts.
        check_can_move(leaving_dock_ok=cm > 0)

        async def run():
            result = await motion.move_by(cm)
            return result.describe(), result.ok
        return await run_to_completion(run(), f"move {cm:+.0f} cm")

    @registry.tool(
        "remember_place",
        "Remember where I am right now under a name ('the couch', 'the door'), so I can go back "
        "there later. Only when someone tells me this spot has a name.",
        {"type": "object", "properties": {"name": {"type": "string", "maxLength": 40}},
         "required": ["name"]})
    async def remember_place(args: dict) -> str:
        name = " ".join((args.get("name") or "").split())
        if not name:
            raise ToolError("a place needs a name")
        if pet.moving:
            raise ToolError("I'm still moving; ask me again once I've stopped")
        # While manual control is armed the robot doesn't update its map
        # pose; it catches up about a second after disarming.
        if motion.armed:
            await motion.disarm()
        since = motion.last_disarmed_at
        if since is not None and time.time() - since < 4:
            await asyncio.sleep(4 - (time.time() - since))
        await vacuum.refresh()
        pose = vacuum.state.pose
        if pose is None:
            raise ToolError("I don't know where I am right now")
        # Kept in the reference map's frame: the robot's map can change frames.
        spot = await pet.frame.to_reference(pose.x, pose.y)
        if spot is None:
            raise ToolError("I can't place myself on my map right now, so I couldn't find it again")
        pet.db.set_place(name, *spot)
        return f"remembered this spot as {name}"

    @registry.tool(
        "go_to_place",
        "Drive to a place I know by name (see places_i_know in get_senses). My base plans its "
        "own route around obstacles. Returns at once; I'm told when I "
        "arrive or if I couldn't get there.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    async def go_to_place(args: dict) -> str:
        name = args.get("name") or ""
        target = pet.db.place(name)
        if target is None:
            known = ", ".join(pet.db.places()) or "none yet"
            raise ToolError(f"I don't know a place called {name} (I know: {known})")
        check_can_move(leaving_dock_ok=True)
        target = await pet.frame.to_current(*target)
        if target is None:
            raise ToolError("I can't match my map to the one I remember places on right now")

        async def run():
            await motion.disarm()
            await vacuum.go_to(*target)
            return await await_arrival(vacuum, name, MapPose(*target), cfg.arrive_cm)
        pet.start_motion(run(), f"go to {name}")
        return f"on my way to {name}"


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
        "must be in view, facing me, fairly close. Takes about ten seconds: halfway, I ask them "
        "to take a step back (that's said for me). Tell them to hold still before calling it; "
        "the result says whether it worked.",
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
                 "face_stored": db.face_count(p.id) > 0,
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
                "face_stored": db.face_count(person.id) > 0,
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


# look_direction's relative turns: to the robot's left (+1) or right, up (+1) or down.
LOOK_DIRECTIONS = {"left": (1, 0), "right": (-1, 0), "up": (0, 1), "down": (0, -1)}


def _look_direction_help(cal: CalibrationConfig, tilt_min: float, tilt_max: float,
                         turn_deg: float = 30.0, tilt_deg: float = 15.0, hold_s: float = 20.0) -> str:
    """look_direction's description: which way the angles go comes from the calibration."""
    pan_left, pan_right = (180, 0) if cal.pan_sign > 0 else (0, 180)
    tilt_up, tilt_down = (tilt_min, tilt_max) if cal.tilt_deg_per_elevation_deg > 0 else (tilt_max, tilt_min)
    up_word = "lower" if cal.tilt_deg_per_elevation_deg > 0 else "higher"
    down_word = "higher" if up_word == "lower" else "lower"
    return (f"Point my head. direction: left or right turns it {turn_deg:g} degrees from where it "
            f"looks now, up or down {tilt_deg:g}; ahead looks straight ahead and level. Use direction "
            "when asked to look somewhere; pan and tilt only for an exact angle. "
            f"pan: {pan_right} is far right, {cal.pan_forward_deg:g} straight ahead, "
            f"{pan_left} far left. tilt: {cal.tilt_level_deg:g} is level, {up_word} looks up "
            f"({tilt_up:g} is as far up as it goes), {down_word} looks down ({tilt_down:g} at most). "
            f"Face tracking is off while I hold the pose, {hold_s:g} s (longer while I take photos "
            "with look); then my head goes back to where it was and tracking comes back on.")


def _clamp(value: Optional[float], low: float, high: float) -> Optional[float]:
    return None if value is None else max(low, min(high, float(value)))


def _describe_position(cx: float, cal: CalibrationConfig) -> str:
    """
    Horizontal position in the camera frame, in words the model can use.
    Which side of the image is the robot's left follows the calibration
    (spatial/person.py): on 2026-10-05, with the servos remounted, the
    image's right was the robot's right.
    """
    toward_left = -cal.pan_sign * cal.cx_per_pan_deg_sign * (cx - 0.5)
    if toward_left > 0.15:
        return "to my left"
    if toward_left < -0.15:
        return "to my right"
    return "straight ahead"
