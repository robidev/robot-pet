"""Planning turns with thinking on and off: petd --fake with the real brain."""
import asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from petd.app import App                                              # noqa: E402
from petd.config import load_config                                   # noqa: E402
from petd.events import BrainToolCall, SentenceReady, TurnEnded, TurnFirstText, TurnStarted   # noqa: E402
from petd.log import setup_logging                                    # noqa: E402

REQUESTS = [
    "GLaDOS, go to the kitchen and tell me if anyone is there.",
    "GLaDOS, how much battery do you have, and is that enough to go to the hallway and come back?",
    "GLaDOS, turn around and look behind you, then tell me what you see.",
    "GLaDOS, remember my face, I'm Sam.",
    "GLaDOS, come over here to me.",
    "GLaDOS, drive forward about half a meter, then turn to face your dock.",
]


async def run(thinking: bool) -> list:
    cfg = load_config(ROOT / "local_config.yaml")
    cfg.brain.thinking = thinking
    cfg.brain.runtime_dir = "runtime/llm-bench/brain"
    cfg.api.port = 8799 if thinking else 8798     # the first run's server may linger
    cfg.speaker.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    app.db.set_place("the kitchen", 2800, 2300)
    app.db.set_place("the hallway", 2300, 2700)
    sub = app.bus.subscribe(TurnStarted, TurnFirstText, BrainToolCall, SentenceReady, TurnEnded)
    await asyncio.sleep(6)                    # prestarted, as when petd has been up a while
    out = []
    try:
        for text in REQUESTS:
            app.hear(text)
            row = {"request": text, "turns": 0, "tools": [], "said": [], "first_word_s": None,
                   "turn_s": 0.0, "cost": 0.0}
            began, started, quiet_since = time.monotonic(), None, None
            while time.monotonic() - began < 240:
                try:
                    e = await asyncio.wait_for(sub.get(), 0.5)
                except asyncio.TimeoutError:
                    e = None
                if isinstance(e, TurnStarted):
                    started = time.monotonic()
                    row["turns"] += 1
                    if e.kind != "heard":
                        row["said"].append(f"<{e.kind}: {e.text}>")
                elif isinstance(e, TurnFirstText) and row["first_word_s"] is None:
                    row["first_word_s"] = round(time.monotonic() - started, 2)
                elif isinstance(e, BrainToolCall):
                    row["tools"].append(f"{e.name}({json.dumps(e.arguments)})")
                elif isinstance(e, SentenceReady):
                    row["said"].append(e.text)
                elif isinstance(e, TurnEnded):
                    row["turn_s"] += round(e.duration_s, 1)
                    row["cost"] += e.cost_usd or 0.0
                idle = (not app.brain.busy and app.brain._queue.empty()
                        and (app.motion_task is None or app.motion_task.done()) and row["turns"])
                quiet_since = (quiet_since or time.monotonic()) if idle else None
                if quiet_since and time.monotonic() - quiet_since > 5:
                    break
            else:
                row["said"].append("<gave up after 240 s>")
            out.append(row)
            print(f"[thinking {'on' if thinking else 'off'}] {text}\n   {row['turns']} turn(s), first word "
                  f"{row['first_word_s']} s, turns {row['turn_s']:.1f} s, ${row['cost']:.3f}\n"
                  f"   tools: {row['tools']}\n   said: {' '.join(row['said'])}", flush=True)
    finally:
        await app.close()
    return out


async def main():
    if "--off" in sys.argv:
        results = {"off": await run(False)}
        (Path(__file__).parent / "planning_off.json").write_text(json.dumps(results, indent=1))
        return
    if "--one" in sys.argv:
        REQUESTS[1:] = []
        results = {"off": await run(False)}
    else:
        results = {"on": await run(True), "off": await run(False)}
    (Path(__file__).parent / "planning.json").write_text(json.dumps(results, indent=1))

setup_logging(sys.argv[1] if len(sys.argv) > 1 else "ERROR")
asyncio.run(main())
