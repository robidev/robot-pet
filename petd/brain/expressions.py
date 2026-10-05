"""
Inline actions -> eye and head movement.

A first pass with single poses; cluster E replaces this with keyframe
sequences loaded from memory/emotions.yaml (blinks, nods, breathing).

Eye coordinates: x/y are -1..1 (the eye's gaze offset), aperture 0..1.5
(how wide open it is).

The brain fires expressions and forgets them (fire()): speech never waits
for the head. Each face call is an HTTP request, and with the face offline
one takes ~3 s to fail (2026-09-28), so an [emote] held the next sentence
back ~6 s and a [nod] ~15 s.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Optional, Set

from ..config import CalibrationConfig
from .tags import Action

if TYPE_CHECKING:
    from ..io.face import FaceAdapter

log = logging.getLogger(__name__)

# emote -> (x, y, aperture)
EMOTIONS: dict[str, tuple[float, float, float]] = {
    "neutral":   (0.0,  0.0,  1.0),
    "happy":     (0.0,  0.1,  1.2),
    "curious":   (0.25, 0.15, 1.3),
    "surprised": (0.0,  0.2,  1.5),
    "sleepy":    (0.0, -0.3,  0.35),
    "sad":       (0.0, -0.4,  0.6),
    "thinking":  (-0.4, 0.35, 0.8),
    "annoyed":   (0.15, 0.0,  0.5),
    "love":      (0.0,  0.0,  1.35),
    "listening": (0.0,  0.05, 1.25),
}

# look -> (x, y) gaze, and the head's turn in degrees to the robot's left and
# up. Which way pan and tilt go for that is the calibration's pan_sign and
# tilt_deg_per_elevation_deg: the servos were remounted reversed on
# 2026-10-05, and the angles turned round with them.
GLANCES: dict[str, tuple[float, float, float, float]] = {
    "left":  (-0.8, 0.0, 25.0, 0.0),
    "right": (0.8, 0.0, -25.0, 0.0),
    "up":    (0.0, 0.7, 0.0, -15.0),
    "down":  (0.0, -0.7, 0.0, 15.0),
    "ahead": (0.0, 0.0, 0.0, 0.0),
    "away":  (-0.7, 0.4, 0.0, 0.0),
}
GLANCE_HOLD_S = 1.5         # a glance holds this long, then the head turns back


class Expressions:
    """Applies Action pieces from the reply stream to the face."""

    def __init__(self, face: "FaceAdapter", cal: Optional[CalibrationConfig] = None,
                 pan_centre: float = 90.0, tilt_centre: float = 90.0):
        self.face = face
        self.cal = cal or CalibrationConfig()
        self.tilt_min = getattr(face.cfg, "tilt_min_deg", 0.0)
        self.tilt_max = getattr(face.cfg, "tilt_max_deg", 180.0)
        self.pan_centre = pan_centre
        self.tilt_centre = tilt_centre
        self._resume: Optional[asyncio.Task] = None     # the head back after a glance
        self._return_to: Optional[tuple[Optional[str], Optional[float], Optional[float]]] = None
        self._last: Optional[asyncio.Task] = None       # the newest fired expression
        self._tasks: Set[asyncio.Task] = set()

    def fire(self, action: Action) -> None:
        """
        Applies `action` in the background and returns at once; failures are
        logged and forgotten. Fired expressions run one at a time, in order:
        a nod mustn't interleave with a glance, and the eye's rest after a
        reply must come after the reply's emotes. While the face is
        unreachable they're skipped, not queued up behind each other.
        """
        if not self.face.state.reachable:
            log.debug("face unreachable: skipping %s", action)
            return
        previous = self._last

        async def run() -> None:
            if previous is not None and not previous.done():
                await asyncio.wait([previous])
            await self.apply(action)

        task = asyncio.create_task(run(), name=f"expression:{action.kind}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._last = task

    async def apply(self, action: Action) -> None:
        log.info("expresses %s%s", action.kind, f":{action.value}" if action.value else "")
        try:
            if action.kind == "emote":
                await self.emote(action.value or "neutral")
            elif action.kind == "look":
                await self.glance(action.value or "ahead")
            elif action.kind == "nod":
                await self.nod()
            elif action.kind == "shake":
                await self.shake()
            elif action.kind == "pause":
                await asyncio.sleep(0.4)
            elif action.kind == "rest":
                await self.rest()
            else:
                log.debug("unknown action tag: %s", action)
        except Exception:  # noqa: BLE001 - expression must never break speech
            log.exception("failed to apply %s", action)

    async def emote(self, name: str) -> None:
        x, y, aperture = EMOTIONS.get(name, EMOTIONS["neutral"])
        # The device's idle/tracking behaviour would override a manual
        # target within a few ms, so switch the eye to manual first.
        await self.face.set_eye_mode("manual")
        await self.face.set_eye(x, y, aperture)

    async def glance(self, direction: str) -> None:
        x, y, left_deg, up_deg = GLANCES.get(direction, GLANCES["ahead"])
        await self.face.set_eye_mode("manual")
        await self.face.set_eye(x, y, 1.1)
        if left_deg or up_deg:
            mode, pan, tilt = self._where_to_return()
            pan_offset = self.cal.pan_sign * left_deg
            tilt_offset = -math.copysign(up_deg, self.cal.tilt_deg_per_elevation_deg)
            await self.face.set_servo(mode="manual")
            await self.face.set_servo(
                pan_deg=max(0.0, min(180.0, (pan if pan is not None else self.pan_centre) + pan_offset)),
                tilt_deg=max(self.tilt_min, min(self.tilt_max,
                                                (tilt if tilt is not None else self.tilt_centre) + tilt_offset)))
            self._return_after(GLANCE_HOLD_S, mode, pan, tilt)

    def _where_to_return(self) -> tuple[Optional[str], Optional[float], Optional[float]]:
        """
        The head's mode and pose before this gesture. A glance still holding
        counts as not having happened: its turn back is cancelled, and the
        pose it would have gone back to is the one to return to.
        """
        if self._resume is not None and not self._resume.done():
            self._resume.cancel()
            return self._return_to
        state = self.face.state
        return state.servo_mode, state.pan_deg, state.tilt_deg

    def _return_after(self, delay_s: float, mode: Optional[str], pan: Optional[float],
                      tilt: Optional[float]) -> None:
        """
        After a glance, the head back where it was, and tracking back on if
        it was: without holding up the reply. Tracking alone only follows a
        face in frame, so a glance away from the only face left the head
        looking at nothing for 2.5 minutes (2026-10-05).
        """
        async def later() -> None:
            await asyncio.sleep(delay_s)
            try:
                if pan is not None or tilt is not None:
                    await self.face.set_servo(pan_deg=pan, tilt_deg=tilt)
                if mode == "track":
                    await self.face.set_servo(mode="track")
            except Exception:  # noqa: BLE001 - the next gesture or init tries again
                log.debug("could not turn the head back", exc_info=True)
        if self._resume is not None:
            self._resume.cancel()
        self._return_to = (mode, pan, tilt)
        self._resume = asyncio.create_task(later(), name="glance-back")

    async def nod(self) -> None:
        await self._wiggle(tilt=True)

    async def shake(self) -> None:
        await self._wiggle(tilt=False)

    async def _wiggle(self, tilt: bool) -> None:
        during_glance = self._resume is not None and not self._resume.done()
        mode, pan_back, tilt_back = self._where_to_return()
        state = self.face.state
        pan = state.pan_deg if state.pan_deg is not None else self.pan_centre
        tilt_deg = state.tilt_deg if state.tilt_deg is not None else self.tilt_centre
        await self.face.set_servo(mode="manual")
        for offset in (12.0, -12.0, 0.0):
            if tilt:
                await self.face.set_servo(tilt_deg=max(self.tilt_min, min(self.tilt_max, tilt_deg + offset)))
            else:
                await self.face.set_servo(pan_deg=max(0.0, min(180.0, pan + offset)))
            await asyncio.sleep(0.22)
        if during_glance:               # nodded mid-glance: the glance still turns back after
            self._return_after(GLANCE_HOLD_S, mode, pan_back, tilt_back)
        elif mode == "track":           # a nod mustn't switch tracking off for good
            await self.face.set_servo(mode="track")

    async def rest(self) -> None:
        """Hands the eye back to the device's own idle/tracking behaviour."""
        try:
            await self.face.set_eye_mode("auto")
        except Exception:  # noqa: BLE001
            log.debug("could not return the eye to auto", exc_info=True)
