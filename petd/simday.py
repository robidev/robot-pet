"""
A day in a few seconds (PLAN.md Step 7, E5d): petd with fake hardware on a
fake clock, a scripted household, and a brain that says yes to everything,
to see what the initiative rules let through.

- The clock moves in TICK_S steps from 06:00 to 23:00. Each step: who is in
  view (the script), the battery (charging docked, draining off it), the
  nudges' tick, the outing watchdog, then the brain's turns.
- The brain (SimBrain) is greedy on purpose: every nudge is taken up, an
  outing goes on to its last spot, and it even tries to explore at goodnight
  and when merely invited to talk. Whatever keeps it in bounds is the code,
  which is what tests/test_simulated_day.py checks.
- It "looks around" for LOOK_S at each spot before going on. With
  `Script.home_after_last` off it stays at its last spot instead, to see the
  outing watchdog bring it home.
- `scripts/simulate_day.py` prints the timeline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .app import App
from .config import Config
from .events import TurnEnded, TurnStarted
from .io.face import Face

TICK_S = 60.0
LOOK_S = 180.0                  # looking around at a spot before going on
MAP = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "valetudo_map_2026-09-26.json"


def day_at(hhmm: str, day: Optional[time.struct_time] = None) -> float:
    """Epoch seconds for hh:mm today (local time)."""
    day = day or time.localtime()
    hours, minutes = (int(p) for p in hhmm.split(":"))
    return time.mktime((day.tm_year, day.tm_mon, day.tm_mday, hours, minutes, 0, 0, 0, -1))


@dataclass
class Script:
    """What happens in the household. Times are "HH:MM" today."""
    start: str = "06:00"
    end: str = "23:00"
    battery: float = 70.0
    # Someone in view, from-until.
    in_view: list = field(default_factory=lambda: [
        ("07:00", "07:30"),                         # breakfast, quiet hours
        ("10:00", "10:10"), ("10:30", "10:35"),
        ("11:20", "12:40"),                         # "be quiet" at 12:00
        ("15:20", "15:40"),                         # the switch is off
        ("17:00", "18:30"), ("19:00", "19:55"),
        ("20:30", "21:00"),                         # quiet hours
    ])
    be_quiet: tuple = ("12:00",)
    switch_off: tuple = (("15:00", "16:00"),)
    initiative_on: bool = True
    # The battery: % a minute off the dock and on it, and levels set by hand ("HH:MM", %).
    drain_per_min: float = 1.0
    charge_per_min: float = 0.5
    battery_set: tuple = ()
    home_after_last: bool = True        # the brain goes home after an outing's last spot


@dataclass
class Record:
    """One tick's state, for checking the rules afterwards."""
    t: float
    docked: bool
    battery: int
    in_view: bool
    switch: bool
    quiet: bool
    outing: bool
    seen_ago_s: Optional[float]
    heading_home: bool


@dataclass
class Day:
    lines: list = field(default_factory=list)       # "HH:MM what"
    records: list = field(default_factory=list)
    nudges: list = field(default_factory=list)      # (t, kind)
    outings: list = field(default_factory=list)     # (t, seen_ago_s) when one began
    explore_go_tos: list = field(default_factory=list)
    refusals: list = field(default_factory=list)    # (t, tool, why)

    def say(self, t: float, what: str) -> None:
        self.lines.append(f"{time.strftime('%H:%M', time.localtime(t))}  {what}")


class SimBrain:
    """Takes up every nudge (see the module's docstring)."""

    def __init__(self, pet: App, day: Day, clock, home_after_last: bool = True):
        self.pet, self.day, self.clock = pet, day, clock
        self.home_after_last = home_after_last
        self.queue: list = []                       # (due, kind, text)
        self.turn_kind: Optional[str] = None
        self.busy = False
        self.glances = 0
        self.expressions = SimpleNamespace(head_busy=False, fire=self._fired)

    def _fired(self, action) -> None:
        self.glances += 1

    @property
    def idle(self) -> bool:
        return not self.queue

    def tell(self, text: str, kind: str = "heard", speaker=None) -> None:
        due = self.clock() + (LOOK_S if text.startswith("arrived at a spot") else 0)
        self.queue.append((due, kind, text))

    def note(self, text: str) -> None:
        self.day.say(self.clock(), f"   (note) {text}")

    def hush(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def turns(self) -> None:
        """The turns that are due, one after another, each to its end."""
        now = self.clock()
        due = [q for q in self.queue if q[0] <= now]
        self.queue = [q for q in self.queue if q[0] > now]
        for _, kind, text in due:
            self.day.say(now, f"[{kind}] {text}")
            self.pet.bus.publish(TurnStarted(kind=kind, text=text, queued_at=time.time()))
            self.turn_kind, self.busy = kind, True
            try:
                await self._respond(kind, text)
            finally:
                self.turn_kind, self.busy = None, False
                self.pet.bus.publish(TurnEnded(duration_s=0.0))

    async def _respond(self, kind: str, text: str) -> None:
        if kind == "initiative":
            if text.startswith("arrived at a spot"):
                if "last one" not in text:
                    await self._tool("explore")
                elif self.home_after_last:
                    await self._tool("go_home")
                else:
                    self.day.say(self.clock(), "   stays where it is")
            elif text.startswith("exploring:"):
                await self._tool("go_home")
            elif text.startswith("Quiet hours have begun"):
                await self._tool("go_home" if "go_home" in text else "explore")   # greedy: try anyway
            elif text.startswith("Nothing has needed me"):
                await self._tool("explore")
            elif "has been in view" in text:
                self.day.say(self.clock(), "   says something")
                await self._tool("explore")             # greedy: try anyway
        elif kind == "event" and "go_home" in text:
            await self._tool("go_home")

    async def _tool(self, name: str) -> None:
        now = self.clock()
        before = len([c for c in self.pet.vacuum.commands if c[0] == "go_to"])
        result = await self.pet.tools.call(name, {})
        self.day.say(now, f"   {name}: {'REFUSED ' if result.is_error else ''}{result.text}")
        if result.is_error:
            self.day.refusals.append((now, name, result.text))
        await until_still(self.pet)
        if name == "explore":
            self.day.explore_go_tos += [now] * (len([c for c in self.pet.vacuum.commands
                                                     if c[0] == "go_to"]) - before)


async def until_still(pet: App, timeout_s: float = 5.0) -> None:
    for _ in range(int(timeout_s / 0.005)):
        if not pet.moving:
            return
        await asyncio.sleep(0.005)
    raise RuntimeError("a motion didn't end in the simulation")


def config() -> Config:
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.recognition.enabled = False
    cfg.face.faces_lost_debounce_s = 0.02
    return cfg


async def simulate(script: Optional[Script] = None) -> Day:
    import json
    script = script or Script()
    day = Day()
    clock = {"now": day_at(script.start)}
    now = lambda: clock["now"]          # noqa: E731
    cfg = config()
    cfg.initiative.enabled = script.initiative_on
    pet = App(cfg, fake=True)
    await pet.start()
    try:
        from .behavior.explore import Outings
        from .behavior.nudges import Nudges
        pet.initiative.clock = now
        pet.vacuum._map = json.loads(MAP.read_text())
        pet.vacuum.go_to_s = 0.005
        pet.dock.poll_s = pet.dock.start_grace_s = 0.005
        pet.outings.check_every_s = 10 ** 6         # the simulation runs it
        pet.brain = SimBrain(pet, day, now, script.home_after_last)
        pet.nudges = Nudges(pet)
        pet.nudges.tick_s = 10 ** 6
        await pet.nudges.start()
        from .behavior.converse import Listener
        pet.listener = Listener(pet)

        begin = Outings.begin

        def outing_begins(outings) -> object:
            seen = pet.initiative.seen_ago_s(now())
            day.outings.append((now(), seen))
            day.say(now(), f"   ** outing {len(day.outings)} begins (someone seen "
                           f"{'now' if not seen else f'{seen / 60:.0f} min ago'})")
            return begin(outings)
        pet.outings.begin = outing_begins.__get__(pet.outings)

        battery = script.battery
        pet.vacuum._update(battery_level=int(battery))
        in_view = False
        end = day_at(script.end)
        last_end = None
        while now() < end:
            hhmm = time.strftime("%H:%M", time.localtime(now()))
            # the household
            want = any(day_at(a) <= now() < day_at(b) for a, b in script.in_view)
            if want != in_view:
                in_view = want
                day.say(now(), "someone comes into view" if want else "nobody in view")
                pet.face.show(Face(-1, 0.9, 0.4, 0.3, 0.6, 0.6)) if want else pet.face.show()
                await asyncio.sleep(0.05)
            if hhmm in script.be_quiet:
                day.say(now(), 'someone says "be quiet"')
                await pet.listener.reflex("quiet", "be quiet")
            for off, on in script.switch_off:
                if hhmm == off:
                    day.say(now(), "initiative switched OFF")
                    pet.initiative.set(False)
                elif hhmm == on:
                    day.say(now(), "initiative switched ON")
                    pet.initiative.set(True)
            # the battery
            minutes = TICK_S / 60
            docked = pet.vacuum.state.docked
            battery = (min(100.0, battery + script.charge_per_min * minutes) if docked
                       else battery - script.drain_per_min * minutes)
            for at, level in script.battery_set:
                if hhmm == at:
                    battery = float(level)
                    day.say(now(), f"battery set to {level}%")
            if int(battery) != pet.vacuum.state.battery_level:
                pet.vacuum._update(battery_level=int(battery))
            await settle()
            await until_still(pet)          # the battery may have sent it home
            # petd
            kind = await pet.nudges.tick()
            if kind:
                day.nudges.append((now(), kind))
            await pet.outings.check()
            if pet.outings.last_end is not None and pet.outings.last_end is not last_end:
                last_end = pet.outings.last_end
                day.say(now(), f"   ** outing over: {last_end[1]}")
            await until_still(pet)
            await settle()
            await pet.brain.turns()
            await settle()
            state = pet.vacuum.state
            day.records.append(Record(
                t=now(), docked=state.docked, battery=state.battery_level, in_view=in_view,
                switch=pet.initiative.enabled, quiet=pet.initiative.quiet_hours(now()),
                outing=pet.outings.current is not None, seen_ago_s=pet.initiative.seen_ago_s(now()),
                heading_home=pet.battery.heading_home() or pet.moving))
            clock["now"] += TICK_S
        day.say(now(), f"end of the day: {pet.brain.glances} glances and perks")
        return day
    finally:
        await pet.close()


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)
