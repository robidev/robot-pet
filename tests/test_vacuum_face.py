import asyncio
import json
from pathlib import Path

from petd.bus import EventBus
from petd.config import FaceConfig
from petd.events import FacesPresence, VacuumStateChanged
from petd.io.face import Face, FakeFace, parse_status
from petd.io.vacuum import FakeVacuum, MapPose, VacuumState, changed_fields, parse_state

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_state_from_real_robot_fixture():
    map_json = json.loads((FIXTURES / "valetudo_map.json").read_text())
    attrs = json.loads((FIXTURES / "valetudo_attributes.json").read_text())
    state = VacuumState(reachable=True, **parse_state(map_json, attrs))
    assert state.status == "docked" and state.docked and not state.moving
    assert state.battery_level == 100
    assert state.pose == MapPose(2560, 2549, 342)
    assert state.charger == MapPose(2560, 2530)
    assert state.pixel_size == 5


def test_changed_fields_ignores_timestamp():
    a = VacuumState(reachable=True, status="idle", updated=1)
    b = VacuumState(reachable=True, status="moving", updated=2)
    assert changed_fields(a, b) == ("status",)
    assert changed_fields(a, VacuumState(reachable=True, status="idle", updated=3)) == ()


async def test_fake_vacuum_publishes_changes():
    bus = EventBus()
    sub = bus.subscribe(VacuumStateChanged)
    vac = FakeVacuum(bus)
    await vac.go_to(100, 200)
    statuses = []
    while not statuses or statuses[-1] != "idle":
        event = await asyncio.wait_for(sub.get(), 1)
        statuses.append(event.state.status)
    assert statuses == ["docked", "moving", "idle"]
    assert "pose" in event.changed and vac.state.pose.x == 100


def test_face_geometry_and_status_parse():
    f = Face.from_device({"id": 2, "confidence": 0.9,
                          "box": {"left": 0.2, "top": 0.1, "right": 0.4, "bottom": 0.5}})
    assert f.recognized and abs(f.cx - 0.3) < 1e-9 and abs(f.h - 0.4) < 1e-9
    s = parse_status({"face_enabled": True, "audio_configured": False, "enrolled_faces": 1,
                      "servo": {"mode": "track", "pan": 88.0, "tilt": 95.0},
                      "eye": {"x": 0, "y": 0, "aperture": 1, "mode": "auto"},
                      "time": {"synced": True}})
    assert s.reachable and not s.audio_configured and s.pan_deg == 88.0


async def test_presence_is_debounced():
    bus = EventBus()
    face = FakeFace(FaceConfig(faces_lost_debounce_s=0.2), bus)
    await face.start()
    sub = bus.subscribe(FacesPresence)
    someone = Face(-1, 0.9, 0.4, 0.3, 0.6, 0.7)
    face.show(someone)
    assert (await asyncio.wait_for(sub.get(), 1)).present
    face.show()                                  # brief dropout...
    await asyncio.sleep(0.05)
    face.show(someone)                           # ...back before the debounce
    await asyncio.sleep(0.3)
    assert sub.get_nowait() is None
    face.show()
    assert not (await asyncio.wait_for(sub.get(), 1)).present
    await face.close()


async def test_a_missed_poll_or_two_is_not_offline():
    # 2026-09-23 13:08: one missed poll in a WiFi stall told the brain its
    # body was offline, and it refused a command. It came back 3 s later.
    import asyncio

    from petd.bus import EventBus
    from petd.config import VacuumConfig
    from petd.io.vacuum import ValetudoVacuum
    vacuum = ValetudoVacuum(VacuumConfig(offline_after_s=0.2), EventBus())
    answer = (None, [{"__class": "StatusStateAttribute", "value": "idle", "flag": "none"}])

    def stalled():
        raise TimeoutError("read timed out")
    vacuum._fetch = lambda: answer
    assert (await vacuum.refresh()).reachable
    vacuum._fetch = stalled
    state = await vacuum.refresh()
    assert state.reachable and state.status == "idle"      # the last state, kept
    await asyncio.sleep(0.25)
    assert not (await vacuum.refresh()).reachable          # silent too long: offline
    vacuum._fetch = lambda: answer
    assert (await vacuum.refresh()).reachable
