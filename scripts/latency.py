"""
Where a turn's time goes (PLAN.md 4.9, G4), from the event trace:

    .venv/bin/python scripts/latency.py                      # the latest run (runtime/logs/latest)
    .venv/bin/python scripts/latency.py runtime/logs/<run>   # another run
    .venv/bin/python scripts/latency.py --live               # petd's last 300 events (GET /events)

For each turn, from the end of the speech to the pet's first spoken sentence:

    end of speech -> transcript (whisper) -> brain starts (queue) -> model's
    first word (thinking, and tools before it) -> first sentence -> synthesized
    (piper) -> speaking

Each step runs from the previous one, so they add up to the total. What
someone in the room feels is the total, plus the robot's own playback lag
(speaker.playback_latency_s, a configured guess). Turn duration is shown
too, but the rest of a reply is spoken while the model is still writing it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("transcript", "brain starts", "first word", "first sentence", "synthesized", "speaking")


def load_trace(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(events, key=lambda e: e["t"])


def load_live(url: str) -> list[dict]:
    with urllib.request.urlopen(f"{url}/events?n=300", timeout=5) as r:
        return sorted(json.load(r), key=lambda e: e["t"])


def analyse(events: list[dict]) -> list[dict]:
    """One dict per turn: its start, text, steps [(name, seconds)], tools, total, duration."""
    turns = []
    for i, start in enumerate(events):
        if start["type"] != "TurnStarted":
            continue
        rest = events[i + 1:]
        end = next((e for e in rest if e["type"] == "TurnEnded"), None)
        until = end["t"] if end else float("inf")
        within = [e for e in rest if e["t"] <= until]

        marks = []                          # (step name, time it happened)
        heard = next((e for e in reversed(events[:i]) if e["type"] == "Heard"
                      and e["text"] == start["text"] and e["t"] <= start["queued_at"] + 0.5), None)
        if start["kind"] == "heard" and heard is not None:
            origin = ("end of speech", heard["t_end"])
            marks.append(("transcript", heard["t"]))
        else:
            origin = ("handed to the brain", start["queued_at"])
        marks.append(("brain starts", start["t"]))
        first_text = next((e for e in within if e["type"] == "TurnFirstText"), None)
        if first_text:
            marks.append(("first word", first_text["t"]))
        sentence = next((e for e in within if e["type"] == "SentenceReady"), None)
        synth = speaking = None
        if sentence:
            marks.append(("first sentence", sentence["t"]))
            after = [e for e in events[i + 1:] if e["t"] >= sentence["t"] and e["t"] < sentence["t"] + 60]
            synth = next((e for e in after if e["type"] == "SentenceSynthesized"
                          and e["text"] == sentence["text"]), None)
            if synth:
                marks.append(("synthesized", synth["t"]))
                speaking = next((e for e in after if e["type"] == "SpeakingStarted"
                                 and e["t"] >= synth["t"] - 0.001), None)
                if speaking:
                    marks.append(("speaking", speaking["t"]))

        steps, previous = [], origin[1]
        for name, t in marks:
            steps.append((name, t - previous))
            previous = t
        turns.append({
            "t": start["t"], "kind": start["kind"], "text": start["text"], "origin": origin[0],
            "steps": steps, "total": previous - origin[1], "spoke": speaking is not None,
            "tools": [(e["name"], e["duration_s"], e["is_error"]) for e in within if e["type"] == "ToolRan"],
            "synth": (synth["synth_s"], synth["audio_s"]) if synth else None,
            "duration": end["duration_s"] if end else None,
            "cost": end.get("cost_usd") if end else None,
            "sentences": end.get("sentences") if end else None,
        })
    return turns


def describe(turn: dict) -> str:
    when = time.strftime("%H:%M:%S", time.localtime(turn["t"]))
    text = turn["text"] if len(turn["text"]) <= 70 else turn["text"][:67] + "..."
    lines = [f"{when}  {turn['kind']}: {text!r}"]
    for name, seconds in turn["steps"]:
        extra = ""
        if name == "first word" and turn["tools"]:
            extra = "   tools: " + ", ".join(f"{n} {d * 1000:.0f} ms{' (error)' if err else ''}"
                                             for n, d, err in turn["tools"])
        if name == "synthesized" and turn["synth"]:
            extra = f"   piper, {turn['synth'][1]:.1f} s of audio"
        lines.append(f"    -> {name:<15} {seconds:6.2f} s{extra}")
    what = "first spoken sentence" if turn["spoke"] else "last step seen"
    lines.append(f"    =  {what} {turn['total']:.2f} s after the {turn['origin']}")
    if turn["duration"] is not None:
        cost = f", ${turn['cost']:.4f}" if turn["cost"] else ""
        lines.append(f"    turn done in {turn['duration']:.1f} s, {turn['sentences']} sentences{cost}")
    return "\n".join(lines)


def summary(turns: list[dict]) -> str:
    spoken = [t for t in turns if t["kind"] == "heard" and t["spoke"]
              and [s[0] for s in t["steps"]] == list(STEPS)]
    if not spoken:
        return "no complete heard-and-answered turns to summarize"
    lines = [f"median over {len(spoken)} heard-and-answered turns:"]
    for index, name in enumerate(STEPS):
        lines.append(f"    -> {name:<15} {statistics.median(t['steps'][index][1] for t in spoken):6.2f} s")
    lines.append(f"    =  first word     {statistics.median(t['total'] for t in spoken):6.2f} s "
                 "after the end of speech")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=str(ROOT / "runtime" / "logs" / "latest"),
                        help="a run's log folder or its events.jsonl (default: the latest run)")
    parser.add_argument("--live", action="store_true", help="petd's last 300 events instead")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    try:
        events = load_live(args.url) if args.live else load_trace(Path(args.run))
    except (OSError, ValueError) as exc:
        sys.exit(f"no events: {exc}")
    turns = analyse(events)
    if not turns:
        sys.exit("no brain turns in these events")
    print("\n\n".join(describe(t) for t in turns))
    print()
    print(summary(turns))


if __name__ == "__main__":
    main()
