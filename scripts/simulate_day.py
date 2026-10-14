"""
A day of the pet's initiative in a few seconds (PLAN.md Step 7, E5d): petd
with fake hardware on a fake clock, a scripted household and a brain that
says yes to every nudge (petd/simday.py). Prints the timeline: who came and
went, every nudge and what the brain did with it, outings, what was refused
and why.

    .venv/bin/python scripts/simulate_day.py
    .venv/bin/python scripts/simulate_day.py --off      # the switch off all day

tests/test_simulated_day.py checks the same day against the house rules.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vacuum-api"))
sys.path.insert(0, str(ROOT / "face-api"))

from petd.simday import Script, simulate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--off", action="store_true", help="the initiative switch off all day")
    args = parser.parse_args()
    logging.basicConfig(level=logging.CRITICAL)
    day = asyncio.run(simulate(Script(initiative_on=not args.off)))
    print("\n".join(day.lines))
    print(f"\n{len(day.outings)} outings, {len(day.nudges)} nudges, {len(day.refusals)} refusals")


if __name__ == "__main__":
    main()
