import asyncio
from dataclasses import replace

from petd.bus import EventBus
from petd.config import VacuumConfig
from petd.events import VacuumStateChanged
from petd.io.robot_wifi import WlanmgrPause
from petd.io.vacuum import VacuumState


def reachable(bus, up=True):
    bus.publish(VacuumStateChanged(state=VacuumState(reachable=up), changed=("reachable",)))


async def test_the_scans_are_paused_whenever_the_robot_comes_back():
    calls = []

    async def run(argv, timeout_s):
        calls.append(argv)
        return 0, "wlanmgr 967 paused (state T): no roaming scans\nwlanmgr 967 is paused"

    bus = EventBus()
    cfg = replace(VacuumConfig(), ssh_key="~/.ssh/robot_key")
    pauser = WlanmgrPause(cfg, bus, run)
    pauser.start()
    reachable(bus)                          # petd's first poll
    await asyncio.sleep(0.05)
    assert len(calls) == 1 and pauser.paused
    argv = calls[0]
    assert argv[0] == "ssh" and argv[argv.index("-i") + 1].endswith("/.ssh/robot_key")
    assert "BatchMode=yes" in argv and argv[-2] == "root@192.168.101.43"
    assert "wlanmgr_pause.sh stop" in argv[-1] and "/proc/uptime" in argv[-1]

    bus.publish(VacuumStateChanged(state=VacuumState(reachable=True), changed=("battery_level",)))
    reachable(bus, up=False)                # the nightly reboot...
    await asyncio.sleep(0.05)
    assert len(calls) == 1
    reachable(bus)                          # ...and back: stock wlanmgr, paused again
    await asyncio.sleep(0.05)
    assert len(calls) == 2
    await pauser.close()


async def test_a_failed_pause_is_only_logged(caplog):
    async def run(argv, timeout_s):
        return 255, "ssh: connect to host 192.168.101.43 port 22: No route to host"

    bus = EventBus()
    pauser = WlanmgrPause(replace(VacuumConfig(), ssh_key="~/.ssh/robot_key"), bus, run)
    assert await pauser.pause() is False
    assert "could not pause wlanmgr" in caplog.text


async def test_without_a_key_nothing_is_run(caplog):
    async def run(argv, timeout_s):
        raise AssertionError("ssh without a key")

    pauser = WlanmgrPause(VacuumConfig(), EventBus(), run)     # the default: no key, as in git
    assert await pauser.pause() is False
    assert "no vacuum.ssh_key" in caplog.text
