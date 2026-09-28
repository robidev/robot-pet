import asyncio
import json
from dataclasses import replace

from petd.app import App
from petd.config import Config
from petd.events import Heard, SpeakingFinished


async def test_fake_app_echo_loop_and_stop():
    cfg = Config()
    cfg.api.enabled = False
    cfg.speaker.playback_latency_s = 0.0
    app = App(cfg, fake=True, echo=True)
    await app.start()
    try:
        heard = app.bus.subscribe(Heard)
        spoken = app.bus.subscribe(SpeakingFinished)
        app.hear("hello there")
        assert (await asyncio.wait_for(heard.get(), 1)).text == "hello there"
        done = await asyncio.wait_for(spoken.get(), 5)
        assert done.text == "You said: hello there"

        # While speaking, anything "heard" is treated as our own echo.
        app.speaker.say("a fairly long sentence to keep the speaker busy for a bit")
        await asyncio.sleep(0.2)
        app.hear("this is my own voice")
        await asyncio.sleep(0.1)
        assert heard.get_nowait() is None

        await app.stop_everything("test")
        assert ("stop",) in app.vacuum.commands
    finally:
        await app.close()


async def test_senses_without_the_head():
    # 2026-09-28: with the head offline, get_senses failed whole (battery and
    # all) after waiting seconds for the head, with a traceback in the log.
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    try:
        async def gone():
            raise ConnectionError("No route to host")
        app.face.current_faces = gone          # went offline since the last status poll
        result = await app.tools.call("get_senses", {})
        senses = json.loads(result.text)
        assert not result.is_error and senses["head"] == {"reachable": False}
        assert senses["body"]["battery_percent"] is not None

        app.face._state = replace(app.face.state, reachable=False)   # known offline: not asked
        result = await app.tools.call("get_senses", {})
        assert not result.is_error and "offline" in json.loads(result.text)["sees"]
        for tool, args in (("look", {}), ("look_direction", {"pan": 90})):
            result = await app.tools.call(tool, args)
            assert result.is_error and "offline" in result.text
    finally:
        await app.close()
