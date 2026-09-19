"""
Inline actions -> eye and head movement.

A first pass with single poses; cluster E replaces this with keyframe
sequences loaded from memory/emotions.yaml (blinks, nods, breathing).

Eye coordinates: x/y are -1..1 (the eye's gaze offset), aperture 0..1.5
(how wide open it is).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

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

# look -> (x, y) gaze, and the head offset in degrees from centre
GLANCES: dict[str, tuple[float, float, float, float]] = {
    "left":  (-0.8, 0.0, 25.0, 0.0),
    "right": (0.8, 0.0, -25.0, 0.0),
    "up":    (0.0, 0.7, 0.0, 15.0),
    "down":  (0.0, -0.7, 0.0, -15.0),
    "ahead": (0.0, 0.0, 0.0, 0.0),
    "away":  (-0.7, 0.4, 0.0, 0.0),
}


class Expressions:
    """Applies Action pieces from the reply stream to the face."""

    def __init__(self, face: "FaceAdapter", pan_centre: float = 90.0, tilt_centre: float = 90.0):
        self.face = face
        self.pan_centre = pan_centre
        self.tilt_centre = tilt_centre

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
        x, y, pan_offset, tilt_offset = GLANCES.get(direction, GLANCES["ahead"])
        await self.face.set_eye_mode("manual")
        await self.face.set_eye(x, y, 1.1)
        if pan_offset or tilt_offset:
            state = self.face.state
            pan = (state.pan_deg if state.pan_deg is not None else self.pan_centre) + pan_offset
            tilt = (state.tilt_deg if state.tilt_deg is not None else self.tilt_centre) + tilt_offset
            await self.face.set_servo(mode="manual")
            await self.face.set_servo(pan_deg=max(0.0, min(180.0, pan)),
                                      tilt_deg=max(0.0, min(180.0, tilt)))

    async def nod(self) -> None:
        await self._wiggle(tilt=True)

    async def shake(self) -> None:
        await self._wiggle(tilt=False)

    async def _wiggle(self, tilt: bool) -> None:
        state = self.face.state
        pan = state.pan_deg if state.pan_deg is not None else self.pan_centre
        tilt_deg = state.tilt_deg if state.tilt_deg is not None else self.tilt_centre
        await self.face.set_servo(mode="manual")
        for offset in (12.0, -12.0, 0.0):
            if tilt:
                await self.face.set_servo(tilt_deg=max(0.0, min(180.0, tilt_deg + offset)))
            else:
                await self.face.set_servo(pan_deg=max(0.0, min(180.0, pan + offset)))
            await asyncio.sleep(0.22)

    async def listening(self) -> None:
        await self.emote("listening")

    async def thinking(self) -> None:
        await self.emote("thinking")

    async def rest(self) -> None:
        """Hands the eye back to the device's own idle/tracking behaviour."""
        try:
            await self.face.set_eye_mode("auto")
        except Exception:  # noqa: BLE001
            log.debug("could not return the eye to auto", exc_info=True)
