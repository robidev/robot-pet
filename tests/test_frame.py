import json
import math
from pathlib import Path

import pytest

from petd.config import Config, MotionConfig
from petd.io.vacuum import MapPose
from petd.spatial import frame as frame_module
from petd.spatial.frame import MapFrame

FIXTURES = Path(__file__).parent / "fixtures"
FULL = json.loads((FIXTURES / "valetudo_map_2026-09-26.json").read_text())         # the whole room
PARTIAL = json.loads((FIXTURES / "valetudo_map_2026-09-26_partial.json").read_text())  # ~74 deg off
FULL_DOCK, PARTIAL_DOCK = (2558, 2531), (2530, 2551)


class Vacuum:
    def __init__(self, map_json):
        self.last_map = map_json


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "map" / "reference.json"
    path.parent.mkdir()
    path.write_text(json.dumps(FULL))
    return path


async def test_a_map_in_another_frame_is_moved_onto_the_reference(reference):
    frame = MapFrame(Vacuum(PARTIAL), MotionConfig(), reference)
    dock = await frame.to_reference(*PARTIAL_DOCK)
    assert math.dist(dock, FULL_DOCK) < 15
    back = await frame.to_current(*dock)
    assert math.dist(back, PARTIAL_DOCK) < 0.01
    assert math.dist(frame.to_reference_now(*PARTIAL_DOCK), dock) < 0.01


async def test_the_alignment_is_only_searched_again_when_the_map_stops_matching(reference, monkeypatch):
    searches = []
    real = frame_module.align
    monkeypatch.setattr(frame_module, "align", lambda ref, new: searches.append(1) or real(ref, new))
    vacuum = Vacuum(PARTIAL)
    frame = MapFrame(vacuum, MotionConfig(), reference)
    first = await frame.alignment()
    assert await frame.alignment() == first and len(searches) == 1
    vacuum.last_map = FULL                      # a new map, in the reference's own frame
    found = await frame.alignment()
    assert len(searches) == 2 and abs(found.angle_deg) < 1


async def test_the_first_big_enough_map_becomes_the_reference(tmp_path):
    path = tmp_path / "reference.json"
    vacuum = Vacuum(PARTIAL)                    # 15 m2 of the room: too little
    frame = MapFrame(vacuum, MotionConfig(), path)
    assert await frame.alignment() is None and not path.exists()
    vacuum.last_map = FULL
    assert await frame.to_reference(100, 200) == (100, 200)
    assert json.loads(path.read_text()) == FULL
    assert MapFrame(Vacuum(FULL), MotionConfig(), path).reference is not None     # kept for next time


async def test_a_map_that_does_not_match_places_nothing(reference):
    import random
    rng = random.Random(1)
    elsewhere = json.loads(json.dumps(FULL))
    for layer in elsewhere["layers"]:
        if layer["type"] == "wall":             # scattered posts, nothing like the room
            layer["compressedPixels"] = [v for _ in range(300)
                                         for v in (rng.randrange(400, 700), rng.randrange(400, 650), 1)]
    frame = MapFrame(Vacuum(elsewhere), MotionConfig(), reference)
    assert await frame.to_reference(2558, 2531) is None
    assert frame.to_reference_now(2558, 2531) is None


async def test_places_are_stored_in_the_reference_frame_and_driven_to_in_the_current_one(reference):
    from petd.app import App
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.face.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    try:
        app.frame = MapFrame(app.vacuum, cfg.motion, reference)
        app.vacuum._map = PARTIAL
        app.vacuum._update(status="idle", pose=MapPose(*PARTIAL_DOCK))
        result = await app.tools.call("remember_place", {"name": "the dock"})
        assert not result.is_error, result.text
        assert math.dist(app.db.place("the dock"), FULL_DOCK) < 15
        app.vacuum._update(pose=MapPose(2600, 2600))
        result = await app.tools.call("go_to_place", {"name": "the dock"})
        assert not result.is_error, result.text
        await app.motion_task
        go_to = [c for c in app.vacuum.commands if c[0] == "go_to"][-1]
        assert math.dist(go_to[1:], PARTIAL_DOCK) < 0.01
    finally:
        await app.close()
