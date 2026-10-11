"""
Coming over to someone (PLAN.md 4.2 stop-and-look, E4, Step 5).

approach() runs as the one motion (App.start_motion), so "stop" cancels it,
and it steers the head meanwhile (Expressions.claim).

1. A look: tracking centres the face, then the median of a few frames gives
   where the person is (spatial/person.py), from the robot's pose with its
   heading kept across turns on the spot (spatial/heading.py).
2. From further than look_again_m + 0.5 m, a go_to to look_again_m short of
   them first, and a second look from there, also when that go_to stopped
   short: far off the estimate is rough (Step 4: 2.5 m read 1.9 m, taken for
   sitting; Robin on the floor read 4.6 m).
3. A go_to to the free floor standoff_m in front of them (approach_target:
   moved back toward the robot when that's in or near furniture). go_to
   ends facing the way it drove, so roughly toward them; the head tracks.

Which face is the person's: with one face in view, that one; otherwise the
recognizer's track for them (or for whoever is speaking, People.speaker),
else the biggest; then the face nearest the last one, frame to frame. With
tracking on and two people in view, the head may centre the other one; the
face is still picked by position.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import statistics
from typing import TYPE_CHECKING, Optional

from ..io.vacuum import MapPose
from ..spatial.dock import await_arrival
from ..spatial.mapgeo import current_grid
from ..spatial.person import PersonEstimate, approach_target, estimate

if TYPE_CHECKING:
    from ..app import App
    from ..memory.db import Person

log = logging.getLogger(__name__)

LOOK_GAP_S = 0.3            # between the frames of one look
CLOSE_ENOUGH_M = 0.3        # within standoff + this, it stays where it is
LOOK_AGAIN_MARGIN_M = 0.5   # further than look_again_m + this: two legs


def head_claim(pet: "App"):
    """The brain's expressions, which a behaviour claims the head from; None without a brain."""
    brain = pet.brain
    return getattr(brain, "expressions", None) if brain is not None else None


def start_centre(pet: "App", person: Optional["Person"], faces) -> Optional[tuple[float, float]]:
    """Where in the frame to start from: the person's face, as well as it can be told."""
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0].cx, faces[0].cy
    people, recognizer = pet.people, pet.recognizer
    who = person if person is not None else (people.speaker() if people is not None else None)
    if who is not None and recognizer is not None:
        for track in recognizer.tracks:
            if track.person_id == who.id:
                return track.centre
    biggest = max(faces, key=lambda f: f.h)
    return biggest.cx, biggest.cy


async def look(pet: "App", person: Optional["Person"]) -> Optional[PersonEstimate]:
    """Where the person is now: tracking on, a moment to centre, then a few frames."""
    face, cal, pcfg = pet.face, pet.cfg.calibration, pet.cfg.person
    await face.set_servo(mode="track")
    await asyncio.sleep(pcfg.look_settle_s)
    face_z = person.face_z_m if person is not None else None
    seen: list[PersonEstimate] = []
    centre: Optional[tuple[float, float]] = None
    for index in range(max(1, pcfg.look_frames)):
        if index:
            await asyncio.sleep(LOOK_GAP_S)
        frame = await face.current_faces()
        if centre is None:
            centre = start_centre(pet, person, frame.faces)
        if centre is None or not frame.faces:
            continue
        chosen = min(frame.faces, key=lambda f: (f.cx - centre[0]) ** 2 + (f.cy - centre[1]) ** 2)
        centre = chosen.cx, chosen.cy
        e = estimate(chosen, frame.pan_deg, frame.tilt_deg, cal, pcfg, pet.robot_pose(), None, face_z)
        if e.xy is not None:
            seen.append(e)
    if not seen:
        return None
    middle = statistics.median_low([e.distance_m for e in seen])
    chosen = next(e for e in seen if e.distance_m == middle)
    log.info("approach: a look: %s at (%.0f, %.0f) from %d frames",
             chosen.describe(), *chosen.xy, len(seen))
    return chosen


async def drive(pet: "App", goal: tuple[float, float], label: str, arrive_cm: float) -> tuple[str, bool]:
    if pet.motion is not None:
        await pet.motion.disarm()
    await pet.vacuum.go_to(*goal)
    return await await_arrival(pet.vacuum, label, MapPose(*goal), arrive_cm)


def _robot_xy(pet: "App") -> Optional[tuple[float, float]]:
    pose = pet.robot_pose()
    return (pose.x, pose.y) if pose is not None else None


async def approach(pet: "App", person: Optional["Person"]) -> tuple[str, bool]:
    """Over to the person, standoff_m in front of them. (outcome for the brain, ok)."""
    who = person.name if person is not None else "them"
    pcfg, arrive_cm = pet.cfg.person, pet.cfg.motion.arrive_cm
    expressions = head_claim(pet)
    if expressions is not None:
        expressions.claim("approach")
    try:
        seen = await look(pet, person)
        robot = _robot_xy(pet)
        if seen is None or robot is None:
            why = "I can't tell where I am" if robot is None else f"I lost sight of {who}"
            return f"I didn't come over: {why}", False
        grid = await asyncio.to_thread(current_grid, pet.vacuum)
        away_m = math.dist(robot, seen.xy) / 100
        if away_m <= pcfg.standoff_m + CLOSE_ENOUGH_M:
            return f"I stayed put: I'm already about {away_m:.1f} m from {who}", True

        if away_m > pcfg.look_again_m + LOOK_AGAIN_MARGIN_M:
            partway, _ = approach_target(robot, seen.xy, grid, dataclasses.replace(pcfg, standoff_m=pcfg.look_again_m))
            if partway is None:
                return f"I didn't come over: there's no free floor between me and {who}", False
            log.info("approach: %.1f m away; first to (%.0f, %.0f), %.1f m short",
                     away_m, *partway, pcfg.look_again_m)
            outcome, ok = await drive(pet, partway, f"the spot a little short of {who}", 2 * arrive_cm)
            # Stopped short (furniture, or a first look that put them too far:
            # Robin sitting on the floor read 4.6 m, 2026-10-11) is still
            # closer: look again from there. An error or no start is not.
            if not ok and not outcome.startswith("stopped"):
                return f"I couldn't get close to {who}: {outcome}", False
            seen = await look(pet, person)
            robot = _robot_xy(pet)
            if seen is None or robot is None:
                return f"I came part of the way, but then lost sight of {who}", False

        goal, note = approach_target(robot, seen.xy, grid, pcfg)
        if goal is None:
            return (f"I can see {who} about {math.dist(robot, seen.xy) / 100:.1f} m away, but there's no "
                    "free floor in front of them for me"), False
        log.info("approach: to (%.0f, %.0f) in front of %s (%s)", *goal, who, note)
        outcome, ok = await drive(pet, goal, f"the spot in front of {who}", arrive_cm)
        if not ok:
            return f"I couldn't get all the way to {who}: {outcome}", False
        return f"I came over: I'm in front of {who} now, about {pcfg.standoff_m:.1f} m away", True
    finally:
        if expressions is not None:
            expressions.release("approach")
        try:
            await pet.face.set_servo(mode="track")
        except Exception:  # noqa: BLE001 - the head may be gone; the next look tries again
            log.debug("approach: couldn't turn tracking on", exc_info=True)
