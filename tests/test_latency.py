"""G4 (PLAN.md 4.9): timings on the bus, the event trace, and scripts/latency.py reading them back."""

import asyncio
import importlib.util
import json
import time

import pytest

from petd.app import App
from petd.brain.backend import TextDelta, ToolStarted, TurnDone
from petd.brain.brain import Brain
from petd.config import PROJECT_ROOT, Config
from petd.events import Heard, SpeakingFinished, ToolRan
from petd.jsonable import event_record

spec = importlib.util.spec_from_file_location("latency", PROJECT_ROOT / "scripts" / "latency.py")
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


class SlowBackend:
    """Thinks, calls a tool, then answers in two sentences."""

    async def start_episode(self, system_prompt: str) -> None: ...
    async def end_episode(self) -> None: ...

    async def send(self, user_turn: str):
        await asyncio.sleep(0.05)
        yield ToolStarted("get_senses", {})
        await asyncio.sleep(0.02)
        yield TextDelta("Hello there. ")
        await asyncio.sleep(0.02)
        yield TextDelta("Nice to see you.")
        yield TurnDone(cost_usd=0.001)


@pytest.fixture
async def pet(tmp_path):
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = False
    cfg.speaker.playback_latency_s = 0.0
    app = App(cfg, fake=True, run_dir=tmp_path)
    await app.start()
    try:
        yield app
    finally:
        await app.close()


async def test_a_turn_is_timed_from_end_of_speech_to_first_word(pet):
    brain = Brain(pet, SlowBackend())
    await brain.start()
    done = pet.bus.subscribe(SpeakingFinished)
    try:
        now = time.time()
        pet.bus.publish(Heard(text="hello pet", t_start=now - 1.2, t_end=now - 0.2))
        brain.tell("hello pet")
        await asyncio.wait_for(done.get(), 5)
        await asyncio.sleep(0.05)
    finally:
        await brain.close()

    events = [json.loads(json.dumps(event_record(e), default=str)) for e in pet.bus.history]
    turns = latency.analyse(sorted(events, key=lambda e: e["t"]))
    assert len(turns) == 1
    turn = turns[0]
    assert [name for name, _ in turn["steps"]] == list(latency.STEPS)
    assert turn["spoke"] and turn["sentences"] == 2
    assert sum(s for _, s in turn["steps"]) == pytest.approx(turn["total"])
    steps = dict(turn["steps"])
    assert steps["transcript"] == pytest.approx(0.2, abs=0.05)     # t_end -> Heard
    assert steps["first word"] >= 0.07                               # the model's 0.07 s
    assert turn["synth"] is not None and turn["synth"][1] > 0        # piper's audio length
    assert "first spoken sentence" in latency.describe(turn)
    assert "median over 1" in latency.summary(turns)


async def test_tool_runs_are_published(pet):
    ran = pet.bus.subscribe(ToolRan)
    await pet.tools.call("get_senses", {})
    await pet.tools.call("move", {"cm": -5000})
    first, second = await ran.get(), await ran.get()
    assert first.name == "get_senses" and not first.is_error and first.duration_s >= 0
    assert second.name == "move" and second.is_error


async def test_the_run_folder_gets_the_whole_event_trace(tmp_path):
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    app = App(cfg, fake=True, run_dir=tmp_path)
    await app.start()
    app.hear("anyone there?")
    await asyncio.sleep(0.05)
    await app.close()
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(r["type"] == "Heard" and r["text"] == "anyone there?" for r in records)
    assert all("t" in r and "type" in r for r in records)
