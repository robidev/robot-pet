"""
Hardware smoke tests for petd's adapters (PLAN.md cluster B acceptance).

    .venv/bin/python scripts/smoke.py vacuum          # read-only: status/battery/pose changes
    .venv/bin/python scripts/smoke.py face            # face/motion/presence events live
    .venv/bin/python scripts/smoke.py stt             # what the pet hears (speak to the face)
    .venv/bin/python scripts/smoke.py say "Hello."    # plays on the robot (needs socat on :6000)
    .venv/bin/python scripts/smoke.py echo            # hear -> say back; checks the echo gate

None of these move the robot. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from petd.app import App                    # noqa: E402
from petd.config import load_config         # noqa: E402
from petd.events import (                   # noqa: E402
    FacesChanged, FacesPresence, Heard, HeardDropped, MotionDetected, SpeakingFinished,
    SpeakingStarted, SpeechEnded, SpeechStarted, VacuumStateChanged,
)
from petd.log import setup_logging          # noqa: E402

WATCH = {
    "vacuum": (VacuumStateChanged,),
    "face": (FacesChanged, FacesPresence, MotionDetected),
    "stt": (SpeechStarted, SpeechEnded, Heard, HeardDropped),
    "echo": (Heard, HeardDropped, SpeakingStarted, SpeakingFinished),
}
NEEDS = {
    "vacuum": {"vacuum"},
    "face": {"face"},
    "stt": {"face", "stt"},          # face: points its audio at this PC
    "say": {"speaker"},
    "echo": {"face", "stt", "speaker"},
}


def describe(event) -> str:
    if isinstance(event, VacuumStateChanged):
        s = event.state
        return f"vacuum {s.status} battery={s.battery_level}% pose={s.pose} changed={event.changed}"
    if isinstance(event, FacesChanged):
        faces = ", ".join(f"id={f.id} cx={f.cx:.2f} h={f.h:.2f}" for f in event.faces) or "none"
        return f"faces: {faces}  pan={event.pan_deg} tilt={event.tilt_deg} seq={event.seq}"
    return f"{type(event).__name__} {event}"


async def main(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    needed = NEEDS[args.what]
    cfg.vacuum.enabled = "vacuum" in needed
    cfg.face.enabled = "face" in needed
    cfg.stt.enabled = "stt" in needed
    cfg.speaker.enabled = "speaker" in needed
    cfg.api.enabled = False

    app = App(cfg, echo=args.what == "echo")
    sub = app.bus.subscribe(*WATCH.get(args.what, (SpeakingFinished,)))
    await app.start()
    try:
        if args.what == "say":
            utt = app.speaker.say(" ".join(args.text) or "Hello, and welcome to the Aperture Science enrichment center.")
            await utt.wait()
            print("interrupted" if utt.interrupted else "done")
            return
        async for event in sub:
            print(describe(event), flush=True)
    finally:
        await app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=sorted(NEEDS))
    parser.add_argument("text", nargs="*")
    parser.add_argument("--config")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
