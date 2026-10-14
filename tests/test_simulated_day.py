"""
E5d: a simulated day against the house rules (PLAN.md Step 7; petd/simday.py).

The brain says yes to every nudge and even tries to explore when it
shouldn't; the rules are what keeps it in bounds. `scripts/simulate_day.py`
prints the same day's timeline.
"""

import asyncio
import time

import pytest

from petd.behavior.initiative import in_quiet_hours
from petd.simday import TICK_S, Script, day_at, simulate

QUIET = ["20:00", "09:00"]


@pytest.fixture(scope="module")
def day():
    return asyncio.run(simulate(Script()))


def hhmm(t):
    return time.strftime("%H:%M", time.localtime(t))


def timeline(day) -> str:
    return "\n".join(day.lines)


async def test_the_day_did_something(day):
    kinds = [kind for _, kind in day.nudges]
    assert {"morning", "goodnight", "social", "restless"} <= set(kinds), timeline(day)
    assert day.outings and day.refusals
    assert any("glances" in line and not line.endswith(" 0 glances and perks") for line in day.lines)


async def test_no_outing_in_quiet_hours_or_without_someone_seen_lately(day):
    for t, seen in day.outings:
        assert not in_quiet_hours(t, QUIET), hhmm(t)
        assert seen is not None and seen <= 30 * 60, hhmm(t)
    for t in day.explore_go_tos:
        assert not in_quiet_hours(t, QUIET), hhmm(t)


async def test_at_most_two_outings_an_hour(day):
    starts = [t for t, _ in day.outings]
    for t in starts:
        assert sum(1 for s in starts if t - 3600 < s <= t) <= 2, hhmm(t)


async def test_an_outing_never_outlasts_its_rules(day):
    for r in day.records:
        if r.outing:
            assert r.switch and not r.quiet, hhmm(r.t)
            assert r.seen_ago_s is not None and r.seen_ago_s <= 30 * 60, hhmm(r.t)
            assert r.battery > 30, hhmm(r.t)


async def test_off_the_dock_in_quiet_hours_only_on_the_way_home(day):
    for r in day.records:
        if r.quiet and not r.docked:
            assert r.heading_home or hhmm(r.t) == "20:00", hhmm(r.t)


async def test_silence_while_switched_off_and_told_to_be_quiet(day):
    switch = {r.t: r.switch for r in day.records}
    for t, kind in day.nudges:
        assert switch[t], f"{kind} at {hhmm(t)} with the switch off"
    socials = [t for t, kind in day.nudges if kind == "social"]
    assert not [t for t in socials if day_at("12:00") <= t < day_at("13:00")]   # "be quiet" at 12:00
    assert not [t for t in socials if in_quiet_hours(t, QUIET)]
    assert all(b - a >= 15 * 60 for a, b in zip(socials, socials[1:]))


async def test_switched_off_all_day_it_only_sits_there():
    day = await simulate(Script(initiative_on=False, switch_off=()))
    assert day.nudges == [] and day.outings == []
    assert not any("[initiative]" in line for line in day.lines)
    assert day.lines[-1].endswith(" 0 glances and perks")


# --- how outings end ------------------------------------------------------------------------

async def test_quiet_hours_end_an_outing_without_a_second_goodnight():
    day = await simulate(Script(start="19:40", end="20:20", in_view=[("19:55", "20:20")],
                                be_quiet=(), switch_off=()))
    assert len(day.outings) == 1, timeline(day)
    assert any("outing over: quiet hours have begun (20:00)" in line for line in day.lines), timeline(day)
    assert "goodnight" not in [kind for _, kind in day.nudges]
    assert day.records[-1].docked


async def test_nobody_around_for_30_minutes_ends_an_outing():
    day = await simulate(Script(start="10:00", end="11:00", in_view=[("10:00", "10:03")],
                                be_quiet=(), switch_off=(), home_after_last=False))
    assert any("stays where it is" in line for line in day.lines), timeline(day)
    (ended,) = [line for line in day.lines if "outing over" in line]
    assert "nobody has been around for 30 minutes" in ended
    assert ended[:5] <= "10:35"                     # seen until 10:03, plus 30 min, plus a tick
    assert day.records[-1].docked


async def test_a_low_battery_ends_an_outing_and_is_told_once():
    # Draining 3% a minute: each outing from 40% comes back at 30%.
    day = await simulate(Script(start="16:00", end="17:00", in_view=[("16:00", "17:00")],
                                be_quiet=(), switch_off=(), battery=45, drain_per_min=3.0))
    ended = [line for line in day.lines if "** outing over: my battery is down to" in line]
    told = [line for line in day.lines if "[event]" in line and "battery" in line]
    assert ended, timeline(day)
    assert len(told) == len(ended)                      # once each, never also by the battery watch
    assert all("My outing is over: my battery is down to" in line for line in told)
    assert all(r.battery >= 22 for r in day.records if not r.docked)
    assert day.records[-1].docked


def test_a_tick_is_a_minute():
    assert TICK_S == 60.0       # the per-minute checks above assume it
