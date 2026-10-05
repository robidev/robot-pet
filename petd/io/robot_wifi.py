"""
Keeps the robot's WiFi roaming scans paused while petd runs (2026-10-05).

Xiaomi's wlanmgr scans for a better access point every 30 s and leaves the
robot's radio deaf for ~1.45 s each time: the speaker's audio stalls, and so
do commands (PLAN.md, findings of 2026-09-27). `/root/wlanmgr_pause.sh stop`
on the robot pauses it (SIGSTOP) without changing the boot process or
wlanmgr itself, but every reboot starts it stock again: the robot reboots
itself nightly at 21:54, so a pause by hand lasted until that evening.

So this runs the pause over SSH whenever the robot becomes reachable: on
petd's first poll, and after every reboot. Right after a boot wlanmgr is
still bringing the WiFi up, so the robot waits until it has been up
`wlanmgr_settle_s` before pausing. A failure is logged and changes nothing
else; the pause is harmless to repeat.

The SSH key's path comes from the (git-ignored) config, never the repo.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from ..events import VacuumStateChanged

if TYPE_CHECKING:
    from ..bus import EventBus
    from ..config import VacuumConfig

log = logging.getLogger(__name__)

PAUSE_SCRIPT = "/root/wlanmgr_pause.sh"

# Runs a command; returns (exit code, output).
Runner = Callable[[list, float], Awaitable[tuple[int, str]]]


def remote_command(settle_s: float) -> str:
    settle = int(settle_s)
    return (f'up=$(cut -d. -f1 /proc/uptime); '
            f'if [ "$up" -lt {settle} ]; then sleep $(({settle} - up)); fi; '
            f'{PAUSE_SCRIPT} stop && {PAUSE_SCRIPT} status')


async def run_process(argv: list, timeout_s: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, f"no answer within {timeout_s:.0f} s"
    return proc.returncode, out.decode(errors="replace").strip()


class WlanmgrPause:
    def __init__(self, cfg: "VacuumConfig", bus: "EventBus", run: Optional[Runner] = None):
        self.cfg = cfg
        self.bus = bus
        self.run = run or run_process
        self.paused: Optional[bool] = None      # the last attempt's outcome, None before one
        self._sub = None
        self._task: Optional[asyncio.Task] = None
        self._attempt: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Subscribes at once: call before the vacuum's first poll, or its first 'reachable' is missed."""
        self._sub = self.bus.subscribe(VacuumStateChanged)
        self._task = asyncio.create_task(self._watch(), name="wlanmgr-pause")

    async def close(self) -> None:
        for task in (self._task, self._attempt):
            if task is not None:
                task.cancel()
        if self._sub is not None:
            self._sub.close()

    async def _watch(self) -> None:
        async for event in self._sub:
            if "reachable" in event.changed and event.state.reachable:
                if self._attempt is not None and not self._attempt.done():
                    self._attempt.cancel()
                self._attempt = asyncio.create_task(self.pause(), name="wlanmgr-pause-attempt")

    def argv(self) -> list:
        key = os.path.expanduser(self.cfg.ssh_key)
        return ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", "-o", "HostKeyAlgorithms=+ssh-rsa",
                f"root@{self.cfg.host}", remote_command(self.cfg.wlanmgr_settle_s)]

    async def pause(self) -> bool:
        if not self.cfg.ssh_key:
            log.warning("robot WiFi: no vacuum.ssh_key in the config, so wlanmgr's roaming scans "
                        "stay on (the voice may stall every 30 s)")
            self.paused = False
            return False
        try:
            code, out = await self.run(self.argv(), self.cfg.wlanmgr_settle_s + 30.0)
        except Exception as exc:  # noqa: BLE001 - ssh missing, key unreadable, ...
            code, out = -1, str(exc)
        last = out.splitlines()[-1] if out else ""
        self.paused = code == 0 and "is paused" in last
        if self.paused:
            log.info("robot WiFi: %s: no roaming scans while petd runs", last)
        else:
            log.warning("robot WiFi: could not pause wlanmgr (exit %s): %s", code, out or "no output")
        return self.paused
