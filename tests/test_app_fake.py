import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from petd.app import App
from petd.brain.expressions import Expressions
from petd.config import Config
from petd.events import Heard, SpeakingFinished
from petd.io.face import Face


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


async def test_look_left_turns_a_little_not_to_the_end():
    # 2026-10-05: asked to look left, the brain sent pan 0, the servo's end.
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    try:
        await app.face.set_servo(mode="track", pan_deg=75.0, tilt_deg=90.0)
        result = await app.tools.call("look_direction", {"direction": "left"})
        assert not result.is_error and "pan 45" in result.text      # pan grows to the right now
        await app.tools.call("look_direction", {"direction": "up"})
        assert (app.face.state.pan_deg, app.face.state.tilt_deg) == (45.0, 105.0)
        assert app.face.state.servo_mode == "manual"
        await app.tools.call("look_direction", {"direction": "ahead"})
        assert (app.face.state.pan_deg, app.face.state.tilt_deg) == (75.0, 90.0)
        result = await app.tools.call("look_direction", {"direction": "sideways"})
        assert result.is_error
    finally:
        await app.close()


async def test_a_glance_with_look_direction_turns_once_and_stays(monkeypatch):
    # 2026-10-05: "[look:left]" plus look_direction(left) in one reply turned
    # 60 degrees, and the glance's turn-back undid the held pose 1.5 s later.
    monkeypatch.setattr("petd.brain.expressions.GLANCE_HOLD_S", 0.05)
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    try:
        app.brain = SimpleNamespace(expressions=Expressions(app.face, cfg.calibration))
        await app.face.set_servo(mode="track", pan_deg=90.0, tilt_deg=120.0)
        await app.brain.expressions.glance("left")
        assert app.face.state.pan_deg == 60.0
        await app.tools.call("look_direction", {"direction": "left"})
        await asyncio.sleep(0.15)
        assert (app.face.state.pan_deg, app.face.state.tilt_deg) == (60.0, 120.0)
        assert app.face.state.servo_mode == "manual"

        await app.brain.expressions.glance("down")
        await app.tools.call("look_direction", {"direction": "down"})
        await asyncio.sleep(0.15)
        assert app.face.state.tilt_deg == 105.0     # higher tilt looks up since the remount
    finally:
        await app.close()


async def test_a_held_look_goes_back_to_tracking():
    # 2026-10-05: after "look left" tracking stayed off and the head stared on.
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.face.look_hold_s = 0.2
    app = App(cfg, fake=True)
    await app.start()
    try:
        expressions = Expressions(app.face, cfg.calibration)
        app.brain = SimpleNamespace(expressions=expressions)
        await app.face.set_servo(mode="track", pan_deg=90.0, tilt_deg=120.0)
        await app.tools.call("look_direction", {"direction": "left"})
        await app.tools.call("look_direction", {"direction": "up"})   # a second look: back to the first's start
        assert (app.face.state.pan_deg, app.face.state.tilt_deg, app.face.state.servo_mode) == (60.0, 135.0, "manual")

        await asyncio.sleep(0.12)
        await expressions.glance("right")       # the reply's glance moves the eye, not the held head
        await app.tools.call("look", {})        # a photo of the held view: the hold starts over
        await asyncio.sleep(0.12)
        assert (app.face.state.pan_deg, app.face.state.servo_mode) == (60.0, "manual")
        await asyncio.sleep(0.2)
        assert (app.face.state.pan_deg, app.face.state.tilt_deg, app.face.state.servo_mode) == (90.0, 120.0, "track")

        await app.tools.call("look_direction", {"direction": "right"})
        await app.tools.call("track_faces", {"on": False})   # on purpose: no turning back later
        await asyncio.sleep(0.3)
        assert (app.face.state.pan_deg, app.face.state.servo_mode) == (120.0, "manual")
    finally:
        await app.close()


def face(cx=0.5):
    return Face(-1, 0.9, cx - 0.1, 0.3, cx + 0.1, 0.6)


def search_app(monkeypatch, sweeps=2):
    monkeypatch.setattr("petd.brain.expressions.SEARCH_STEP_S", 0.01)
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.face.search_speed_deg_s = 1000.0      # 180 deg in 0.18 s
    cfg.face.search_sweeps = sweeps
    return cfg


async def started(cfg):
    app = App(cfg, fake=True)
    await app.start()
    told = []
    app.brain = SimpleNamespace(expressions=Expressions(app.face, cfg.calibration),
                                tell=lambda text, kind="heard", speaker=None: told.append((kind, text)))
    return app, told


async def test_a_search_stops_on_a_face_and_tracks_it(monkeypatch):
    app, told = await started(search_app(monkeypatch))
    try:
        await app.face.set_servo(mode="manual", pan_deg=60.0, tilt_deg=90.0)
        result = await app.tools.call("search_for_faces", {})
        assert "searching" in result.text
        for _ in range(100):                    # to the nearer end (0) first, then across
            if app.face.state.pan_deg is not None and app.face.state.pan_deg > 100.0:
                break
            await asyncio.sleep(0.005)
        assert app.face.state.tilt_deg == 110.0
        app.face._state = replace(app.face.state, pan_deg=130.0)
        app.face.show(face())                   # seen in a frame taken at pan 130
        app.face._state = replace(app.face.state, pan_deg=150.0)   # the head has moved on
        await asyncio.sleep(0.1)
        assert (app.face.state.pan_deg, app.face.state.servo_mode) == (130.0, "track")
        assert told and told[-1][0] == "event" and "found a face" in told[-1][1]
        assert not app.brain.expressions.searching

        result = await app.tools.call("search_for_faces", {})   # a face in view: no search
        assert "already see a face" in result.text and not app.brain.expressions.searching
    finally:
        await app.close()


async def test_a_search_without_faces_ends_ahead_and_tracking(monkeypatch):
    app, told = await started(search_app(monkeypatch, sweeps=1))
    try:
        await app.face.set_servo(mode="manual", pan_deg=150.0, tilt_deg=90.0)
        expressions = app.brain.expressions
        await app.tools.call("search_for_faces", {})
        await asyncio.sleep(0.05)
        await expressions.glance("left")        # a glance mid-search moves only the eye
        assert expressions.searching and not expressions._glance_pending()
        for _ in range(200):
            if not expressions.searching:
                break
            await asyncio.sleep(0.01)
        assert (app.face.state.pan_deg, app.face.state.servo_mode) == (75.0, "track")
        assert "found nobody" in told[-1][1]

        await app.tools.call("search_for_faces", {})
        await app.tools.call("look_direction", {"direction": "ahead"})   # a look ends the search
        await asyncio.sleep(0.05)
        assert not expressions.searching and app.face.state.servo_mode == "manual"
    finally:
        await app.close()
