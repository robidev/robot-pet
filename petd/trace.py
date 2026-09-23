"""
The run's event trace: every bus event, one JSON line each, in the run's log
folder (runtime/logs/<run>/events.jsonl), with its class under "type".

The bus keeps only the last 300 events (GET /events), which is one turn or
so; this keeps the whole session, for scripts/latency.py and for reading a
run back afterwards (PLAN.md 4.9).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from .bus import EventBus
from .jsonable import event_record

log = logging.getLogger(__name__)


class EventTrace:
    def __init__(self, bus: EventBus, path: Path):
        self.bus = bus
        self.path = path
        self._sub = None
        self._file = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Subscribes at once, so nothing published after this is missed."""
        self._sub = self.bus.subscribe(maxsize=10000)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._task = asyncio.create_task(self._run(), name="event-trace")
        log.info("tracing events to %s", self.path)

    async def _run(self) -> None:
        async for event in self._sub:
            self._write(event)

    def _write(self, event) -> None:
        try:
            self._file.write(json.dumps(event_record(event), default=str) + "\n")
        except Exception:  # noqa: BLE001 - one odd event must not end the trace
            log.debug("could not trace %r", event, exc_info=True)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
        if self._sub is not None:
            while (event := self._sub.get_nowait()) is not None:     # what's still queued
                self._write(event)
            self._sub.close()
        if self._file:
            self._file.close()
