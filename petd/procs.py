"""
Supervised child processes (whisper-udp-stream, piper): restart with
exponential backoff, stdout lines to a callback, stderr to the log,
graceful terminate -> kill on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Callable, Optional, Sequence

from .bus import EventBus
from .events import ProcessStateChanged


class ManagedProcess:
    def __init__(
        self,
        name: str,
        argv: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        on_stdout_line: Optional[Callable[[str], None]] = None,
        bus: Optional[EventBus] = None,
        restart: bool = True,
        min_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        stop_timeout_s: float = 5.0,
    ):
        self.name = name
        self.argv = list(argv)
        self.cwd = cwd
        self.env = env
        self.on_stdout_line = on_stdout_line
        self.bus = bus
        self.restart = restart
        self.min_backoff_s = min_backoff_s
        self.max_backoff_s = max_backoff_s
        self.stop_timeout_s = stop_timeout_s
        self.log = logging.getLogger(f"petd.proc.{name}")
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._supervise(), name=f"proc:{self.name}")

    async def stop(self) -> None:
        self._stopping = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), self.stop_timeout_s)
            except asyncio.TimeoutError:
                self.log.warning("did not exit after SIGTERM, killing")
                proc.kill()
                await proc.wait()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _publish(self, running: bool, returncode: Optional[int] = None) -> None:
        if self.bus is not None:
            self.bus.publish(ProcessStateChanged(name=self.name, running=running, returncode=returncode))

    async def _supervise(self) -> None:
        backoff = self.min_backoff_s
        while not self._stopping:
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                env = None if self.env is None else {**os.environ, **self.env}
                self._proc = await asyncio.create_subprocess_exec(
                    *self.argv, cwd=self.cwd, env=env,
                    # Own session: Ctrl-C in petd's terminal reaches the
                    # whole process group, and children would die before
                    # petd shuts them down in order.
                    start_new_session=True,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                self.log.error("failed to start %s: %s", self.argv[0], exc)
                returncode = None
            else:
                self.log.info("started (pid %d)", self._proc.pid)
                self._publish(True)
                await asyncio.gather(
                    self._pump(self._proc.stdout, self._handle_stdout),
                    self._pump(self._proc.stderr, self._handle_stderr),
                )
                returncode = await self._proc.wait()
                self._publish(False, returncode)
                if not self._stopping:
                    self.log.warning("exited with code %s", returncode)

            if self._stopping or not self.restart:
                return
            # A process that ran for a while earns a fresh, short backoff.
            if loop.time() - started > 60:
                backoff = self.min_backoff_s
            self.log.info("restarting in %.0fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    @staticmethod
    async def _pump(stream: asyncio.StreamReader, handle: Callable[[str], None]) -> None:
        while True:
            raw = await stream.readline()
            if not raw:
                return
            handle(raw.decode("utf-8", errors="replace").rstrip("\r\n"))

    def _handle_stdout(self, line: str) -> None:
        if self.on_stdout_line is None:
            self.log.info("%s", line)
            return
        try:
            self.on_stdout_line(line)
        except Exception:  # noqa: BLE001 - a bad line must not kill the pump
            self.log.exception("error handling stdout line: %r", line)

    def _handle_stderr(self, line: str) -> None:
        if line.strip():
            self.log.debug("%s", line)
