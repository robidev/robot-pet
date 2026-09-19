"""
In-process async pub/sub. Subscribers get their own bounded queue and
filter by event type (isinstance, so subscribing to Event gets everything).

publish() must be called on the event loop thread; background threads
(the face WebSocket, the Player reader) use publish_threadsafe().
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Optional

from .events import Event

log = logging.getLogger(__name__)


class Subscription:
    def __init__(self, bus: "EventBus", types: tuple, maxsize: int):
        self._bus = bus
        self.types = types
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def matches(self, event: Event) -> bool:
        return isinstance(event, self.types)

    def _offer(self, event: Event) -> None:
        if self.queue.full():
            # Slow consumer: drop the oldest so the newest state wins.
            self.queue.get_nowait()
            self.dropped += 1
            if self.dropped in (1, 10, 100) or self.dropped % 1000 == 0:
                log.warning("subscription %s dropped %d events", self.types, self.dropped)
        self.queue.put_nowait(event)

    async def get(self) -> Event:
        return await self.queue.get()

    def get_nowait(self) -> Optional[Event]:
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        self._bus._unsubscribe(self)

    def __aiter__(self):
        return self

    async def __anext__(self) -> Event:
        return await self.queue.get()


class EventBus:
    def __init__(self, history: int = 300):
        self._subs: list[Subscription] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.history: deque = deque(maxlen=history)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, *types: type, maxsize: int = 1000) -> Subscription:
        sub = Subscription(self, types or (Event,), maxsize)
        self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def publish(self, event: Event) -> None:
        self.history.append(event)
        for sub in list(self._subs):
            if sub.matches(event):
                sub._offer(event)

    def publish_threadsafe(self, event: Event) -> None:
        if self._loop is None:
            raise RuntimeError("EventBus.bind_loop() must be called before publish_threadsafe()")
        self._loop.call_soon_threadsafe(self.publish, event)
