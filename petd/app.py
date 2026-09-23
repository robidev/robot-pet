"""
Wires the adapters, the bus and the local API together, and owns startup
and shutdown order. Behaviors and the brain plug in here in later clusters
(see PLAN.md); until then `echo=True` gives a hear -> speak loop for
testing the audio path and the echo gate end to end.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
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
    def __init__(self, cfg: Config, *, fake: bool = False, echo: bool = False,
                 run_dir: Optional[Path] = None):
        self.cfg = cfg
        self.fake = fake
        self.echo = echo
        self.run_dir = run_dir          # this run's log folder (petd/log.py), if any
        self.bus = EventBus()
        self.trace = None
        self.vacuum: Optional[VacuumAdapter] = None
        self.face: Optional[FaceAdapter] = None
        self.speaker: Optional[Speaker] = None
        self.stt = None
        self.db = None
        self.people = None
        self.recognizer = None
        self.motion = None
        self.dock = None
        self.motion_task: Optional[asyncio.Task] = None
        self.tools = None
        self.brain = None
        self.listener = None
        self._piper: Optional[ManagedProcess] = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        cfg, fake = self.cfg, self.fake
        self.bus.bind_loop(asyncio.get_running_loop())
        if self.run_dir is not None and cfg.log.events:
            from .trace import EventTrace
            self.trace = EventTrace(self.bus, self.run_dir / "events.jsonl")
            self.trace.start()

        if cfg.vacuum.enabled:
            self.vacuum = FakeVacuum(self.bus) if fake else ValetudoVacuum(cfg.vacuum, self.bus)
        if cfg.face.enabled:
            self.face = FakeFace(cfg.face, self.bus) if fake else RobotFace(cfg.face, self.bus, cfg.network.pc_ip)
        if cfg.speaker.enabled:
            self.speaker, self._piper = build_speaker(cfg, self.bus, fake)
        if cfg.stt.enabled:
            gate = self.speaker.overlap_fraction if self.speaker else None
            self.stt = FakeStt(cfg, self.bus, gate) if fake else SttAdapter(cfg, self.bus, gate)

        if self._piper:
            self._piper.start()
        for part in (self.vacuum, self.face, self.speaker, self.stt):
            if part is not None:
                await part.start()

        if self.vacuum is not None and cfg.motion.enabled:
            from .spatial.motion import Motion
            if fake:
                odometry = self.vacuum.odometry
            else:
                from .spatial.odometry import PlayerOdometry
                odometry = PlayerOdometry(cfg.vacuum.host, cfg.player.port)
            self.motion = Motion(self.vacuum, odometry, cfg.motion)
            # Connected from the start (it only streams while armed), so the
            # first turn doesn't also wait for a Player handshake.
            start = getattr(odometry, "start", None)
            if start:
                start()

        if cfg.memory.enabled:
            from .memory.db import MemoryDB
            from .memory.people import People
            # --fake gets a throwaway memory so test runs don't meet real people.
            self.db = MemoryDB(":memory:" if fake else cfg.path(cfg.memory.db_path))
            self.people = People(self, self.db)
            await self.people.start()
            self.recognizer = self._build_recognizer()
            if self.recognizer is not None:
                await self.recognizer.start()

        if self.vacuum is not None:
            from .spatial.dock import Dock
            self.dock = Dock(self)
            await self.dock.start()

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
        await self._cancel_motion()
        for part in (self.listener, self.brain, self.recognizer, self.people, self.dock, self.motion, self.stt,
                     self.speaker, self.face, self.vacuum):
            if part is not None:
                try:
                    await part.close()
                except Exception:  # noqa: BLE001 - keep shutting the rest down
                    log.exception("error closing %s", type(part).__name__)
        if self._piper:
            await self._piper.stop()
        if self.db is not None:
            self.db.close()
        if self.trace is not None:
            await self.trace.close()

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
        await self._cancel_motion()
        if self.vacuum:
            try:
                await self.vacuum.stop_motion()
            except Exception:  # noqa: BLE001 - a stop must not fail loudly
                log.exception("vacuum stop failed")

    def start_motion(self, coro, describe: str, report: bool = True) -> asyncio.Task:
        """
        Runs a motion (a turn, a move, a trip to a place) as a task, one at
        a time, so a stop can cancel it. With `report`, the brain hears how
        it went: a success as a note with its next turn, a failure as an
        event of its own. Without, the caller awaits the task's
        (outcome, ok) itself.
        """
        async def run() -> tuple[str, bool]:
            try:
                outcome, ok = await coro
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - becomes something the pet can say
                log.exception("%s failed", describe)
                outcome, ok = f"{describe} failed: {exc}", False
            log.info("motion: %s", outcome)
            if report and self.brain is not None:
                if ok:
                    self.brain.note(outcome)
                else:
                    self.brain.tell(outcome, kind="event")
            return outcome, ok
        self.motion_task = asyncio.create_task(run(), name="motion")
        return self.motion_task

    async def go_home(self) -> asyncio.Task:
        """Back onto the charger, whatever else was under way (spatial/dock.py)."""
        await self._cancel_motion()
        return self.start_motion(self.dock.go_home(), "go home")

    def _build_recognizer(self):
        """Face recognition on the PC, if it's on and its models are there (PLAN.md 4.7)."""
        rc = self.cfg.recognition
        if not rc.enabled or self.face is None:
            return None
        try:
            from .memory.recognition import FakeEngine, Recognizer
            if self.fake:
                engine = FakeEngine()
            else:
                from .vision.faces import FaceEngine
                engine = FaceEngine(self.cfg.path(rc.models_dir), rc.detector_model, rc.recognizer_model)
        except Exception as exc:  # noqa: BLE001 - missing models or packages: run without it
            log.warning("face recognition off: %s", exc)
            return None
        return Recognizer(self, engine)

    @property
    def recognition_on(self) -> bool:
        return self.recognizer is not None

    @property
    def moving(self) -> bool:
        return self.motion_task is not None and not self.motion_task.done()

    async def _cancel_motion(self) -> None:
        task, self.motion_task = self.motion_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self.motion is not None:
            await self.motion.disarm()

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
