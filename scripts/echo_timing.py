"""
How long does the robot's voice outlast what the speaker thinks it sent?
(PLAN.md 4.9: the "first PCM byte -> audible" and "last byte -> silent" legs.)

The echo gate drops what the mic hears while the pet talks, plus
`speaker.playback_latency_s + speaker.gate_tail_s` after the last audio
is due. If the robot's aplay/socat buffering keeps sound coming for
longer than that, the tail of the pet's own sentence is transcribed as
someone talking to it. This speaks a few lines with the gate off, and
compares when STT heard each one with when its audio was due.

    .venv/bin/python scripts/echo_timing.py            # stay quiet while it runs (~40 s)

Nothing moves. The pet says five sentences through the robot's speaker.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from petd.app import App                    # noqa: E402
from petd.config import load_config         # noqa: E402
from petd.events import Heard               # noqa: E402
from petd.log import setup_logging          # noqa: E402

LINES = [
    "Testing. One, two, three.",
    "This is a measurement of my own voice, which I find flattering.",
    "Short one.",
    "The cake is a lie. I repeat, the cake is a lie.",
    "Measurement complete. You may resume making noise.",
]


def speech_pad_s(extra_args: list) -> float:
    """whisper-udp-stream pads each utterance by --speech-pad-ms (300 by default)."""
    args = [str(a) for a in extra_args]
    if "--speech-pad-ms" in args:
        return int(args[args.index("--speech-pad-ms") + 1]) / 1000
    return 0.3


async def main(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    configured = (cfg.speaker.playback_latency_s, cfg.speaker.gate_tail_s)
    # Raw spans: when the audio was due, with no latency guess on top.
    cfg.speaker.playback_latency_s = 0.0
    cfg.speaker.gate_tail_s = 0.0
    cfg.vacuum.enabled = cfg.brain.enabled = cfg.api.enabled = cfg.memory.enabled = False
    pad = speech_pad_s(cfg.stt.extra_args)

    app = App(cfg, fake=args.fake)
    await app.start()
    try:
        app.stt.echo_gate = None            # hear everything, our own voice included
        heard = app.bus.subscribe(Heard)
        print("waiting for the face to stream audio ...")
        await asyncio.sleep(4)

        rows = []
        for line in LINES:
            utt = app.speaker.say(line)
            await utt.wait()
            start, end = app.speaker._spans[-1]
            await asyncio.sleep(args.listen)
            got = []
            while (event := heard.get_nowait()) is not None:
                got.append(event)
            echoes = [e for e in got if e.t_end > start - 1.0]
            if not echoes:
                print(f"  (heard nothing of {line!r})")
                continue
            first, last = echoes[0], echoes[-1]
            audible_end = last.t_end - pad
            rows.append((first.t_start - start, audible_end - end, last.t_start - end))
            print(f"  {line!r}\n    heard as {' / '.join(e.text for e in echoes)!r}\n"
                  f"    starts {first.t_start - start:+.2f}s after the first byte was due; "
                  f"sound ends ~{audible_end - end:+.2f}s after the last byte was due; "
                  f"last segment starts {last.t_start - end:+.2f}s after it")
            await asyncio.sleep(0.5)

        if not rows:
            print("\nThe mic heard none of it: check the face is streaming (smoke.py stt).")
            return
        tail_lag = max(r[1] for r in rows)
        seg_after = max(r[2] for r in rows)
        print(f"\nstart lag median {statistics.median(r[0] for r in rows):+.2f}s; "
              f"end lag max {tail_lag:+.2f}s; a segment started up to {seg_after:+.2f}s "
              f"after the audio was due")
        need = max(tail_lag, seg_after) + 0.3
        print(f"configured: playback_latency_s {configured[0]} + gate_tail_s {configured[1]} "
              f"= {sum(configured):.2f}s after the last byte")
        print(f"suggested:  playback_latency_s + gate_tail_s >= {need:.2f}s"
              + ("  (already enough)" if sum(configured) >= need else "  <- raise it"))
    finally:
        await app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--fake", action="store_true", help="dry run without hardware")
    parser.add_argument("--listen", type=float, default=4.0,
                        help="seconds to wait for STT after each line")
    args = parser.parse_args()
    setup_logging("WARNING")
    asyncio.run(main(args))
