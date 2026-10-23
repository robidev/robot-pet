"""
What the pet says after a tool that only confirms what it already said
(2026-09-28: "I already spoke: ..."). petd --fake with the real brain,
thinking off; a fresh brain per try; Robin recognized and in view.

    .venv/bin/python bench/llm-bench/after_tool.py <label> [tries]
"""
import asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from petd.app import App                                              # noqa: E402
from petd.config import load_config                                   # noqa: E402
from petd.events import BrainToolCall, SentenceReady, TurnEnded, TurnStarted   # noqa: E402
from petd.io.face import Face                                         # noqa: E402
from petd.log import setup_logging                                    # noqa: E402

REQUESTS = [
    "I'm going to shut you down now.",
    "Go back to your dock, please.",
    "Point your head straight ahead.",
]


PORTS = iter(range(8700, 8790))


async def trial(text: str) -> dict:
    cfg = load_config(ROOT / "local_config.yaml")
    cfg.brain.runtime_dir = "runtime/llm-bench/brain"
    cfg.api.port = next(PORTS)              # a closed app's server may linger
    cfg.speaker.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    try:
        robin = app.db.add_person("Robin")
        app.face.show(Face(-1, 0.9, 0.5, 0.4, 0.3, 0.8))
        app.people.recognized(robin.id, time.time(), 0.84)
        greeting = []
        sub = app.bus.subscribe(TurnStarted, BrainToolCall, SentenceReady, TurnEnded)
        await asyncio.sleep(6)
        quiet = 0.0                             # let the greeting for Robin finish first
        while quiet < 3.0:
            e = sub.get_nowait()
            if isinstance(e, SentenceReady):
                greeting.append(e.text)
            busy = app.brain.busy or not app.brain._queue.empty()
            quiet = 0.0 if busy or e is not None else quiet + 0.1
            await asyncio.sleep(0.1)
        app.brain.tell(text, kind="heard", speaker="Robin")
        before, tools, after, mine = [], [], [], False
        while True:
            e = await asyncio.wait_for(sub.get(), 120)
            if isinstance(e, TurnStarted):
                mine = e.text == text
                continue
            if not mine:
                continue
            if isinstance(e, BrainToolCall):
                tools.append(e.name)
            elif isinstance(e, SentenceReady):
                (after if tools else before).append(e.text)
            elif isinstance(e, TurnEnded):
                return {"request": text, "greeting": greeting, "before": before, "tools": tools,
                        "after": after}
    finally:
        await app.close()


async def main(label: str, tries: int):
    rows = []
    for text in REQUESTS:
        for _ in range(tries):
            row = await trial(text)
            rows.append(row)
            print(f"{text!r}  (greeting: {' '.join(row['greeting'])!r})\n   said {' '.join(row['before'])!r}  tools {row['tools']}"
                  f"\n   after the tool: {' '.join(row['after'])!r}", flush=True)
    out = Path(__file__).parent / f"after_tool_{label}.json"
    out.write_text(json.dumps(rows, indent=1))
    spoke_after = sum(bool(r["after"]) for r in rows if r["tools"])
    print(f"\n{label}: {spoke_after} of {sum(bool(r['tools']) for r in rows)} tool turns said more after the tool")

if __name__ == "__main__":
    setup_logging("ERROR")
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
