"""
Wires the adapters, the bus and the local API together, and owns startup
and shutdown order. Behaviors and the brain plug in here in later clusters
(see PLAN.md); until then `echo=True` gives a hear -> speak loop for
testing the audio path and the echo gate end to end.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .bus import EventBus
from .config import Config
from .events import Heard, StopRequested
from .io.face import FakeFace, FaceAdapter, RobotFace
from .io.speaker import Speaker, build_speaker
from .io.stt import FakeStt, SttAdapter
from .io.vacuum import FakeVacuum, VacuumAdapter, ValetudoVacuum
from .procs import ManagedProcess

log = logging.getLogger(__name__)


class App:
    def __init__(self, cfg: Config, *, fake: bool = False, echo: bool = False):
        self.cfg = cfg
        self.fake = fake
        self.echo = echo
        self.bus = EventBus()
        self.vacuum: Optional[VacuumAdapter] = None
        self.face: Optional[FaceAdapter] = None
        self.speaker: Optional[Speaker] = None
        self.stt = None
        self.db = None
        self.people = None
        self.tools = None
        self.brain = None
        self.listener = None
        self._piper: Optional[ManagedProcess] = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        cfg, fake = self.cfg, self.fake
        self.bus.bind_loop(asyncio.get_running_loop())

        if cfg.vacuum.enabled:
            self.vacuum = FakeVacuum(self.bus) if fake else ValetudoVacuum(cfg.vacuum, self.bus)
        if cfg.face.enabled:
            self.face = FakeFace(cfg.face, self.bus) if fake else RobotFace(cfg.face, self.bus, cfg.network.pc_ip)
        if cfg.speaker.enabled:
            self.speaker, self._piper = build_speaker(cfg, self.bus, fake)
        if cfg.stt.enabled:
            gate = self.speaker.overlaps if self.speaker else None
            self.stt = FakeStt(cfg, self.bus, gate) if fake else SttAdapter(cfg, self.bus, gate)

        if self._piper:
            self._piper.start()
        for part in (self.vacuum, self.face, self.speaker, self.stt):
            if part is not None:
                await part.start()

        if cfg.memory.enabled:
            from .memory.db import MemoryDB
            from .memory.people import People
            # --fake gets a throwaway memory so test runs don't meet real people.
            self.db = MemoryDB(":memory:" if fake else cfg.path(cfg.memory.db_path))
            self.people = People(self, self.db)
            await self.people.start()

        from .brain.tools import build_registry
        self.tools = build_registry(self)
        if not self.echo:
            # The brain and the echo loop both answer Heard events; --echo
            # is the hardware test path, so it wins when both are asked for.
            from .brain.brain import build_brain
            self.brain = build_brain(self)
            if self.brain is not None:
                await self.brain.start()
                from .behavior.converse import Listener
                self.listener = Listener(self)
                await self.listener.start()
                if self.people is not None:
                    self.people.set_notify(lambda text: self.brain.tell(text, kind="event"))

        if cfg.api.enabled:
            from .api.server import serve
            self._tasks.append(asyncio.create_task(serve(self), name="api"))
        if self.echo and self.speaker:
            # Subscribe now, not inside the task, so nothing published
            # before the task first runs is missed.
            sub = self.bus.subscribe(Heard)
            self._tasks.append(asyncio.create_task(self._echo_loop(sub), name="echo"))
        log.info("petd up (%s)", "fake hardware" if fake else "real hardware")

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Reverse of start: stop listening/speaking before letting go of hardware.
        for part in (self.listener, self.brain, self.people, self.stt, self.speaker, self.face, self.vacuum):
            if part is not None:
                try:
                    await part.close()
                except Exception:  # noqa: BLE001 - keep shutting the rest down
                    log.exception("error closing %s", type(part).__name__)
        if self._piper:
            await self._piper.stop()
        if self.db is not None:
            self.db.close()

    async def run_until_stopped(self) -> None:
        await self._stopped.wait()

    def request_shutdown(self) -> None:
        self._stopped.set()

    async def stop_everything(self, source: str) -> None:
        """The big red button: silence and halt. Never raises."""
        log.warning("STOP requested by %s", source)
        self.bus.publish(StopRequested(source=source))
        if self.speaker:
            self.speaker.interrupt()
        if self.vacuum:
            try:
                await self.vacuum.stop_motion()
            except Exception:  # noqa: BLE001 - a stop must not fail loudly
                log.exception("vacuum stop failed")

    def hear(self, text: str, source: str = "api") -> None:
        """Injects text as if it had been heard (goes through the same filters)."""
        if isinstance(self.stt, FakeStt):
            self.stt.inject(text, source=source)
        elif self.stt is not None:
            import time
            now = time.time()
            self.stt.handle_text(text, now, now, source=source)
        else:
            import time
            self.bus.publish(Heard(text=text, t_start=time.time(), t_end=time.time(), source=source))

    async def _echo_loop(self, sub) -> None:
        async for event in sub:
            self.speaker.say(f"You said: {event.text}")
