import asyncio
import json
from pathlib import Path

import httpx
import pytest

from petd.api.server import create_app
from petd.app import App
from petd.config import Config
from petd.events import PersonArrived
from petd.memory.recognition import sample
from petd.vision.matching import normalized, to_blob

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
async def pet():
    cfg = Config()
    cfg.api.enabled = False
    cfg.brain.enabled = False
    cfg.stt.enabled = False
    cfg.speaker.enabled = False
    cfg.recognition.attempt_every_s = 0.01
    cfg.recognition.recheck_every_s = 0
    app = App(cfg, fake=True)
    await app.start()
    try:
        yield app
    finally:
        await app.close()


def client(pet) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(pet)), base_url="http://petd")


async def test_the_dashboard_has_its_panels(pet):
    async with client(pet) as c:
        page = (await c.get("/")).text
    for panel in ("Status", "People", "Latest attempts", "Map", "Events", "Tools"):
        assert panel in page


async def test_people_shows_who_is_in_view_and_what_recognition_did(pet):
    from petd.io.face import Face
    robin = pet.db.add_person("Robin")
    face = normalized(__import__("numpy").random.default_rng(3).normal(size=128))
    for _ in range(5):
        pet.db.add_face_embedding(robin.id, to_blob(face), source="enroll")
    pet.recognizer.reload()
    pet.recognizer.engine.default = [sample(face)]
    arrived = pet.bus.subscribe(PersonArrived)
    pet.face.show(Face(-1, 0.9, 0.4, 0.3, 0.6, 0.6))
    await asyncio.wait_for(arrived.get(), 2)
    async with client(pet) as c:
        people = (await c.get("/people")).json()
    assert people["present"] == ["Robin"]
    assert people["recognition"]["tracks"][0]["name"] == "Robin"
    assert people["recognition"]["tracks"][0]["last_similarity"] > 0.9
    assert people["attempts"] == []         # --fake keeps no crops


async def test_the_map_shows_places_and_sightings(pet):
    async with client(pet) as c:
        assert (await c.get("/map.png")).status_code == 404       # no map in --fake
        pet.vacuum._map = json.loads((FIXTURES / "valetudo_map_2026-09-26.json").read_text())
        pet.db.set_place("kitchen", 2600.0, 2600.0)
        response = await c.get("/map.png?scale=2")
    assert response.status_code == 200 and response.content[:8] == b"\x89PNG\r\n\x1a\n"


async def test_only_attempt_files_are_served(pet, tmp_path):
    from petd.vision.kept import FaceKeeper
    pet.recognizer.keeper = FaceKeeper(tmp_path, 10, str)
    (tmp_path / "secret.jpg").write_bytes(b"x")
    good = "20260927-101722.870_as-Robin_best-Robin-0.61.jpg"
    (tmp_path / "attempts" / good).write_bytes(b"\xff\xd8jpeg")
    async with client(pet) as c:
        assert (await c.get(f"/faces/attempt/{good}")).content == b"\xff\xd8jpeg"
        assert (await c.get("/faces/attempt/..%2Fsecret.jpg")).status_code == 404
        assert (await c.get("/faces/attempt/secret.jpg")).status_code == 404


async def test_the_tool_console_lists_the_tools(pet):
    async with client(pet) as c:
        response = await c.get("/tools")
    if pet.tools is None:
        assert response.status_code == 503
    else:
        assert all({"name", "description", "schema"} <= set(t) for t in response.json())
