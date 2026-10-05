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
import time
from typing import TYPE_CHECKING, Callable, Optional, Set

from ..config import CalibrationConfig
from ..events import FacesChanged
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

# look -> (x, y) gaze, and which way the head turns: to the robot's left (+1)
# or right (-1), up (+1) or down (-1), by face.look_turn_deg / look_tilt_deg.
GLANCES: dict[str, tuple[float, float, int, int]] = {
    "left":  (-0.8, 0.0, 1, 0),
    "right": (0.8, 0.0, -1, 0),
    "up":    (0.0, 0.7, 0, 1),
    "down":  (0.0, -0.7, 0, -1),
    "ahead": (0.0, 0.0, 0, 0),
    "away":  (-0.7, 0.4, 0, 0),
}
GLANCE_HOLD_S = 1.5         # a glance holds this long, then the head turns back
# A search moves the head in steps this far apart (the firmware jumps to a
# manual pose): 10 deg at 20 deg/s. Jumpy is fine (Robin), and a head
# standing still between steps gives the detector sharper frames.
SEARCH_STEP_S = 0.5
# A look asked for (look_direction) holds face.look_hold_s, then turns back
# the same way: without it, the head stared at the held pose until someone
# turned tracking back on (2026-10-05).


def head_offset(left_deg: float, up_deg: float, cal: CalibrationConfig) -> tuple[float, float]:
    """
    Degrees to the robot's left and up, as pan and tilt offsets. Which way
    they go is the calibration's pan_sign and tilt_deg_per_elevation_deg:
    the servos were remounted reversed on 2026-10-05, and the angles turned
    round with them.
    """
    return cal.pan_sign * left_deg, -up_deg * math.copysign(1.0, cal.tilt_deg_per_elevation_deg)


class Expressions:
    """Applies Action pieces from the reply stream to the face."""

    def __init__(self, face: "FaceAdapter", cal: Optional[CalibrationConfig] = None):
        self.face = face
        self.cal = cal or CalibrationConfig()
        self.tilt_min = getattr(face.cfg, "tilt_min_deg", 0.0)
        self.tilt_max = getattr(face.cfg, "tilt_max_deg", 180.0)
        self.turn_deg = getattr(face.cfg, "look_turn_deg", 30.0)
        self.tilt_deg = getattr(face.cfg, "look_tilt_deg", 15.0)
        self.pan_centre = self.cal.pan_forward_deg        # where the head is, if its pose isn't known
        self.tilt_centre = self.cal.tilt_level_deg
        self._resume: Optional[asyncio.Task] = None     # the head back after a glance
        self._return_to: Optional[tuple[Optional[str], Optional[float], Optional[float]]] = None
        self._hold: Optional[asyncio.Task] = None       # the head back after a held look
        self._hold_origin: Optional[tuple[Optional[str], Optional[float], Optional[float]]] = None
        self._hold_s = 0.0
        self._search: Optional[asyncio.Task] = None     # "look for me"
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
        x, y, left, up = GLANCES.get(direction, GLANCES["ahead"])
        await self.face.set_eye_mode("manual")
        await self.face.set_eye(x, y, 1.1)
        if (left or up) and not self.head_busy:      # a held look or a search keeps the head; the eye glances
            mode, pan, tilt = self._where_to_return()
            pan_offset, tilt_offset = head_offset(left * self.turn_deg, up * self.tilt_deg, self.cal)
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

    # --- a held look (look_direction) ---------------------------------------------

    @property
    def holding(self) -> bool:
        return self._hold is not None and not self._hold.done()

    @property
    def searching(self) -> bool:
        return self._search is not None and not self._search.done()

    @property
    def head_busy(self) -> bool:
        return self.holding or self.searching

    def _glance_pending(self) -> bool:
        return self._resume is not None and not self._resume.done()

    def base_pose(self) -> tuple[Optional[float], Optional[float]]:
        """
        Where a look turns from: where the head is, or where it was before a
        glance still holding. A reply says "[look:left]" and calls
        look_direction(left) together; turning from the glance's pose went
        twice as far (2026-10-05).
        """
        if self._glance_pending():
            _, pan, tilt = self._return_to
            return pan, tilt
        return self.face.state.pan_deg, self.face.state.tilt_deg

    async def hold(self, pan: Optional[float], tilt: Optional[float], hold_s: float) -> None:
        """
        Turns the head to (pan, tilt) with tracking off, for hold_s; then back
        to where it was before the first of these looks, and tracking back on
        if it was on. A glance still holding is ended, and its pose is the
        one to go back to.
        """
        self.stop_search()
        if self.holding:
            self._hold.cancel()
            origin = self._hold_origin
        else:
            origin = self._where_to_return()
            if self._glance_pending():
                self._resume.cancel()
        await self.face.set_servo(mode="manual")
        await self.face.set_servo(pan_deg=pan, tilt_deg=tilt)
        self._hold_origin, self._hold_s = origin, hold_s
        self._hold = asyncio.create_task(self._release_after(hold_s), name="look-held")

    def extend_hold(self) -> None:
        """Still looking (a photo of the held view): the hold starts over."""
        if self.holding:
            self._hold.cancel()
            self._hold = asyncio.create_task(self._release_after(self._hold_s), name="look-held")

    def end_hold(self) -> None:
        """Tracking switched on or off on purpose: the head stays as it's told."""
        self.stop_search()
        if self.holding:
            self._hold.cancel()
        if self._glance_pending():
            self._resume.cancel()

    async def _release_after(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        mode, pan, tilt = self._hold_origin
        log.info("the held look is over: back to pan %s, tilt %s%s", pan, tilt,
                 ", tracking" if mode == "track" else "")
        try:
            if pan is not None or tilt is not None:
                await self.face.set_servo(pan_deg=pan, tilt_deg=tilt)
            if mode == "track":
                await self.face.set_servo(mode="track")
        except Exception:  # noqa: BLE001 - the next look or init tries again
            log.debug("could not turn the head back after a held look", exc_info=True)

    # --- "look for me" ----------------------------------------------------------

    def start_search(self, report: Optional[Callable[[str, bool], None]] = None) -> None:
        """
        Pans the head slowly around the room until a face shows up, then
        back to where the frame with it was taken (the head has moved on
        since, ~1 s of detection) and tracking on. report() gets what came
        of it, and whether it found a face. A held look or a glance ends; a
        new search starts over.
        """
        self.stop_search()
        if self.holding:
            self._hold.cancel()
        if self._glance_pending():
            self._resume.cancel()
        self._search = asyncio.create_task(self._run_search(report), name="face-search")

    def stop_search(self) -> None:
        if self.searching:
            self._search.cancel()

    async def _run_search(self, report: Optional[Callable[[str, bool], None]]) -> None:
        cfg = self.face.cfg
        low, high = cfg.search_pan_min_deg, cfg.search_pan_max_deg
        tilt = max(self.tilt_min, min(self.tilt_max, cfg.search_tilt_deg))
        speed = max(1.0, cfg.search_speed_deg_s)
        pan = self.face.state.pan_deg if self.face.state.pan_deg is not None else self.pan_centre
        pan = max(low, min(high, pan))
        near, far = (low, high) if pan - low <= high - pan else (high, low)
        legs = [near] + [far, near] * max(1, cfg.search_sweeps)
        sub = self.face.bus.subscribe(FacesChanged)
        started = time.monotonic()
        found: Optional[FacesChanged] = None
        log.info("searching for a face: pan %g..%g at tilt %g, %g deg/s", low, high, tilt, speed)
        try:
            await self.face.set_servo(mode="manual")
            await self.face.set_servo(pan_deg=pan, tilt_deg=tilt)
            for target in legs:
                found = await self._pan_to(pan, target, speed, sub)
                if found is not None:
                    break
                pan = target
            if found is not None:
                await self.face.set_servo(pan_deg=found.pan_deg, tilt_deg=found.tilt_deg)
                await self.face.set_servo(mode="track")
                log.info("search: a face at pan %.1f, tilt %.1f after %.0f s; tracking",
                         found.pan_deg, found.tilt_deg, time.monotonic() - started)
                outcome = (f"My search found a face after {time.monotonic() - started:.0f} s; "
                           "I'm following it now.")
            else:
                await self.face.set_servo(pan_deg=self.pan_centre, tilt_deg=tilt)
                await self.face.set_servo(mode="track")
                log.info("search: no face after %.0f s", time.monotonic() - started)
                outcome = (f"My search found nobody: I looked around the room {max(1, cfg.search_sweeps)} "
                           "times. I'm looking ahead again, following any face that shows up.")
        except asyncio.CancelledError:
            log.info("search stopped")
            raise
        except Exception:  # noqa: BLE001 - a lost head mustn't take the brain down
            log.exception("search failed")
            outcome = "My search broke off: my head stopped answering."
        finally:
            sub.close()
        if report is not None:
            report(outcome, found is not None)

    async def _pan_to(self, start: float, target: float, speed: float, sub) -> Optional[FacesChanged]:
        """Pans from start to target at speed; the first frame with a face, if one comes."""
        began = time.monotonic()
        span = abs(target - start)
        while True:
            for event in iter(sub.get_nowait, None):
                if event.faces:
                    return event
            done = min(span, speed * (time.monotonic() - began))
            pan = start + math.copysign(done, target - start)
            await self.face.set_servo(pan_deg=pan)
            if done >= span:
                return None
            await asyncio.sleep(SEARCH_STEP_S)

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
