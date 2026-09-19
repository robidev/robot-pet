import asyncio

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
