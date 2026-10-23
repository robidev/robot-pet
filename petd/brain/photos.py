"""
A photo from the head's camera with what someone says, now and then, so the
brain sees who it's talking to without being asked to look (2026-10-23:
"can you see the gesture I'm making?" got "I can't see your hands" with no
photo taken).

- Only with a turn of someone talking (kind "heard"), only with someone in
  view, and at most every `brain.photo_every_s` (the first such turn of a
  conversation always gets one): it stays in the conversation's context.
- No added wait, mostly: when someone starts talking and a photo is due,
  the snapshot is taken then (0.7-1.2 s on 2026-10-23), while they're still
  speaking and the recognizer transcribes. A photo the recognizer or `look`
  took moments ago will do as well.
- Measured on the claude CLI with Haiku (2026-10-23): a 640x480 frame is
  ~510 more input tokens on its turn, ~$0.0004, and the first words came
  no later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

# A photo already taken this recently (by the recognizer, look, a prefetch) will do.
FRESH_S = 3.0
# How long a turn waits for a photo before going without one.
WAIT_S = 2.0


@dataclass(frozen=True)
class TurnPhoto:
    jpeg: bytes
    taken_at: float
    pan_deg: Optional[float]
    tilt_deg: Optional[float]


class TurnPhotos:
    def __init__(self, pet: "App"):
        self.pet = pet
        self.every_s = pet.cfg.brain.photo_every_s
        self._last_sent: Optional[float] = None     # this conversation's last photo
        self._pending: Optional[asyncio.Task] = None

    def new_conversation(self) -> None:
        """A fresh context has no photo in it yet."""
        self._last_sent = None

    def due(self, now: Optional[float] = None) -> bool:
        face = self.pet.face
        if self.every_s <= 0 or face is None or not face.state.reachable or not face.presence.present:
            return False
        return self._last_sent is None or (now or time.time()) - self._last_sent >= self.every_s

    def prefetch(self) -> None:
        """Someone started talking: take the photo now if the turn will want one."""
        if not self.due() or self._fresh() is not None:
            return
        if self._pending is None or self._pending.done():
            self._pending = asyncio.create_task(self._snapshot(), name="turn-photo")

    async def take(self, kind: str) -> Optional[TurnPhoto]:
        """The photo for this turn, or None: not due, nobody in view, or the head didn't answer."""
        if kind != "heard" or not self.due():
            return None
        photo = self._fresh()
        if photo is None:
            task, self._pending = self._pending, None
            if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
                task = asyncio.create_task(self._snapshot(), name="turn-photo")
            try:
                await asyncio.wait_for(asyncio.shield(task), WAIT_S)
            except Exception:  # noqa: BLE001 - a turn without a photo is fine
                log.debug("turn photo: none in time", exc_info=True)
            photo = self._fresh()
        if photo is not None:
            self._last_sent = time.time()
        return photo

    def _fresh(self) -> Optional[TurnPhoto]:
        last = getattr(self.pet.face, "last_snapshot", None)
        if last is None:
            return None
        taken_at, snap = last
        if time.time() - taken_at > FRESH_S or not snap.jpeg:
            return None
        return TurnPhoto(snap.jpeg, taken_at, snap.servo_pan_deg, snap.servo_tilt_deg)

    async def _snapshot(self) -> None:
        await self.pet.face.snapshot()      # it keeps it as face.last_snapshot
