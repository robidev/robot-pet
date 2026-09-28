import asyncio
import time

import numpy as np
import pytest

from petd.app import App
from petd.brain.backend import TextDelta, TurnDone
from petd.brain.brain import Brain
from petd.brain.prompt import build_system_prompt, build_turn, clean_summary
from petd.config import Config
from petd.events import PersonArrived, PersonLeft
from petd.io.face import Face
from petd.memory.db import MemoryDB
from petd.memory.recognition import sample
from petd.vision.matching import normalized, to_blob

RNG = np.random.default_rng(7)


def face(id=-1, cx=0.5):
    return Face(id, 0.9, cx - 0.1, 0.3, cx + 0.1, 0.6)


def someone() -> np.ndarray:
    """A made-up person: a random direction in fingerprint space (unrelated ones score ~0)."""
    return normalized(RNG.normal(size=128))


def view(person: np.ndarray, noise: float = 0.5) -> np.ndarray:
    """One look at them: ~0.9 similar to the person, ~0.8 to another look."""
    wobble = RNG.normal(size=128)
    return normalized(person + wobble * noise / np.linalg.norm(wobble))


def store_face(pet, name: str, person: np.ndarray, n: int = 5):
    """Someone already enrolled, with n fingerprints."""
    record = pet.db.add_person(name)
    for _ in range(n):
        pet.db.add_face_embedding(record.id, to_blob(view(person)), source="enroll")
    pet.recognizer.reload()
    return record


@pytest.fixture
async def pet():
    cfg = Config()
    cfg.api.enabled = False
    cfg.brain.enabled = False
    cfg.stt.enabled = False
    cfg.speaker.enabled = False
    cfg.face.faces_lost_debounce_s = 0.1
    cfg.recognition.attempt_every_s = 0.01
    cfg.recognition.recheck_every_s = 0     # tests that want it turn it on
    cfg.recognition.enroll_timeout_s = 0.3
    cfg.recognition.enroll_step_back_s = 0.0
    app = App(cfg, fake=True)
    await app.start()
    app.recognizer.collect_every_s = 0.0
    told: list[str] = []
    app.people.set_notify(told.append)
    app.told = told
    try:
        yield app
    finally:
        await app.close()


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


# --- the database ------------------------------------------------------------

def test_people_facts_and_familiarity():
    db = MemoryDB(":memory:")
    robin = db.add_person("Robin", face_slot=3)
    assert db.person_by_name("robin").id == robin.id
    assert db.person_by_slot(3).name == "Robin"
    db.set_nickname(robin.id, "Test Subject Two")
    assert db.person_by_name("test subject two").id == robin.id

    db.add_fact("likes coffee", about=robin.id)
    db.add_fact("the plant by the window is fake")
    assert [f.text for f in db.facts(about=robin.id)] == ["likes coffee"]
    assert [f.text for f in db.facts()] == ["the plant by the window is fake"]

    db.add_sighting(robin.id, time.time())
    thresholds = [(1, 1), (3, 2)]
    assert db.count_interaction(robin.id, thresholds).familiarity == 1
    for _ in range(5):
        person = db.count_interaction(robin.id, thresholds)
    assert person.familiarity == 1          # enough interactions, but only one day seen
    db.add_sighting(robin.id, time.time() - 3 * 86400)
    assert db.count_interaction(robin.id, thresholds).familiarity == 2

    db.delete_person(robin.id)
    assert db.person_by_slot(3) is None and db.facts(about=robin.id) == []


def test_notes_are_capped_and_conversations_journal():
    db = MemoryDB(":memory:")
    person = db.add_person("Noah")
    for _ in range(100):
        db.add_note(person.id, "a fairly long note about nothing much.")
    assert len(db.person(person.id).notes) <= 1000

    first = db.start_conversation(t=1000)
    db.add_utterance(first, "Noah", "hello")
    db.end_conversation(first, "Met Noah.")
    db.end_conversation(db.start_conversation(t=2000), None)   # nothing said: no entry
    third = db.start_conversation(t=3000)
    db.end_conversation(third, "Noah came back.")
    assert [c.summary for c in db.journal()] == ["Met Noah.", "Noah came back."]
    assert db.utterances(first) == [("Noah", "hello")]


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "pet.db"
    MemoryDB(path).add_person("Robin")
    assert MemoryDB(path).person_by_name("Robin") is not None


# --- enrollment and forgetting -------------------------------------------------

async def test_enroll_stores_fingerprints_and_skips_the_greeting(pet):
    noah = someone()
    pet.recognizer.engine.frames = [[sample(view(noah))] for _ in range(10)]
    result = await pet.people.enroll("Noah")
    assert "Noah" in result and "5 views" in result
    person = pet.db.person_by_name("Noah")
    assert pet.db.face_count(person.id) == 5 and pet.recognizer.knows_face(person.id)
    assert pet.people.who_is_here() == (["Noah"], 0)

    # Still in view afterwards: known, and no greeting mid-conversation.
    pet.recognizer.engine.default = [sample(view(noah))]
    pet.face.show(face())
    await asyncio.sleep(0.1)
    assert pet.told == []


async def test_enroll_asks_for_a_step_back_and_takes_a_second_set(pet):
    said = []

    class Voice:
        def say(self, text):
            said.append(text)
    pet.speaker = Voice()
    noah = someone()
    pet.recognizer.engine.frames = ([[sample(view(noah), height=190)] for _ in range(5)]
                                    + [[sample(view(noah), height=75)] for _ in range(5)])
    result = await pet.people.enroll("Noah")
    assert "10 views" in result and "step back" in said[0] and len(said) == 1


async def test_enroll_asks_again_and_keeps_only_real_step_backs(pet):
    said = []

    class Voice:
        def say(self, text):
            said.append(text)
    pet.speaker = Voice()
    noah = someone()
    close = [[sample(view(noah), height=190)] for _ in range(5)]
    # First ask: still in place. Second: mid-step (180 px), then really back (90 px).
    pet.recognizer.engine.frames = (close + [[sample(view(noah), height=188)] for _ in range(5)]
                                    + [[sample(view(noah), height=h)] for h in (180, 90, 88, 86, 85)])
    result = await pet.people.enroll("Noah")
    assert len(said) == 2 and "further back" in said[1]
    assert "9 views" in result                        # 5 close + the 4 really back
    assert "didn't step back" not in result


async def test_enroll_without_a_step_back_keeps_the_close_set_only(pet):
    said = []

    class Voice:
        def say(self, text):
            said.append(text)
    pet.speaker = Voice()
    pet.recognizer.engine.default = [sample(view(someone()), height=190)]
    result = await pet.people.enroll("Noah")
    assert len(said) == 2 and "5 views" in result and "didn't step back" in result
    assert pet.db.face_count(pet.db.person_by_name("Noah").id) == 5


async def test_enroll_refuses_nobody_crowds_small_faces_and_lookalikes(pet):
    engine = pet.recognizer.engine
    with pytest.raises(Exception, match="can't see a face"):
        await pet.people.enroll("Noah")

    engine.default = [sample(view(someone()), centre=(0.3, 0.4)),
                      sample(view(someone()), centre=(0.7, 0.4))]
    with pytest.raises(Exception, match="more than one face"):
        await pet.people.enroll("Noah")

    engine.default = [sample(view(someone()), height=30)]
    with pytest.raises(Exception, match="too small"):
        await pet.people.enroll("Noah")

    robin = someone()
    store_face(pet, "Robin", robin)
    engine.default = [sample(view(robin))]
    with pytest.raises(Exception, match="looks like Robin"):
        await pet.people.enroll("Noah")
    assert pet.db.person_by_name("Noah") is None

    # ...unless the model insists, after the person does.
    await pet.people.enroll("Noah", insist=True)
    assert pet.db.face_count(pet.db.person_by_name("Noah").id) == 5


async def test_a_second_enrollment_needs_insist_and_replaces_the_face(pet):
    noah = someone()
    person = store_face(pet, "Noah", noah, n=3)
    pet.recognizer.engine.default = [sample(view(noah))]
    with pytest.raises(Exception, match="already have Noah's face"):
        await pet.people.enroll("noah")
    await pet.people.enroll("Noah", insist=True)
    assert pet.db.face_count(person.id) == 5


async def test_forget_deletes_the_fingerprints_with_the_person(pet):
    person = store_face(pet, "Noah", someone())
    result = await pet.people.forget("noah")
    assert "Noah" in result
    assert pet.db.face_embeddings() == {} and pet.db.person_by_name("Noah") is None
    assert not pet.recognizer.knows_face(person.id)
    with pytest.raises(Exception, match="don't know anyone"):
        await pet.people.forget("Noah")


# --- recognition, sticky identity and greetings ------------------------------------

async def test_a_known_face_is_named_greeted_once_and_sticks(pet):
    robin_face = someone()
    robin = store_face(pet, "Robin", robin_face)
    arrived = pet.bus.subscribe(PersonArrived)
    left = pet.bus.subscribe(PersonLeft)
    pet.recognizer.engine.default = [sample(view(robin_face))]

    pet.face.show(face())
    assert (await asyncio.wait_for(arrived.get(), 2)).name == "Robin"
    await settle()
    assert len(pet.told) == 1 and pet.told[0].startswith("Robin just came into view")

    pet.face.show(face(), face(cx=0.8))     # someone else joins
    await settle()
    assert pet.people.who_is_here() == (["Robin"], 1)
    assert pet.people.sole_person() is None

    pet.face.show()                         # only nobody-in-view ends it
    assert (await asyncio.wait_for(left.get(), 1)).name == "Robin"
    assert pet.people.who_is_here() == ([], 0)

    pet.face.show(face())                   # back within greet_every_h: no second greeting
    assert (await asyncio.wait_for(arrived.get(), 2)).person_id == robin.id
    await settle()
    assert len(pet.told) == 1


async def test_a_swap_within_the_presence_debounce_is_noticed(pet):
    # 2026-09-27: Robin stepped out and Claudia in during a 1-2 s detection gap;
    # presence never dropped, so she stayed "Robin" for four minutes.
    pet.cfg.face.faces_lost_debounce_s = 5.0
    robin_face, claudia_face = someone(), someone()
    store_face(pet, "Robin", robin_face)
    store_face(pet, "Claudia", claudia_face)
    arrived = pet.bus.subscribe(PersonArrived)
    left = pet.bus.subscribe(PersonLeft)
    pet.recognizer.engine.default = [sample(view(robin_face))]
    pet.face.show(face())
    assert (await asyncio.wait_for(arrived.get(), 2)).name == "Robin"

    pet.face.show()                         # a moment with nobody in view, mid-visit
    pet.recognizer.engine.default = [sample(view(claudia_face))]
    pet.face.show(face())
    assert (await asyncio.wait_for(arrived.get(), 2)).name == "Claudia"
    assert (await asyncio.wait_for(left.get(), 2)).name == "Robin"
    assert pet.people.who_is_here() == (["Claudia"], 0)
    assert pet.people.sole_person().name == "Claudia"

    for _ in range(50):                     # the same person after a flicker stays, quietly
        if not pet.recognizer._visiting():
            break
        await asyncio.sleep(0.01)
    pet.face.show()
    pet.face.show(face())
    await asyncio.sleep(0.2)
    assert not pet.recognizer._visiting()
    assert pet.people.who_is_here() == (["Claudia"], 0)
    assert left.get_nowait() is None and arrived.get_nowait() is None


async def test_a_wrong_name_is_corrected_by_a_later_look(pet):
    # No gap in detection at all: only the periodic look can notice.
    robin_face, claudia_face = someone(), someone()
    store_face(pet, "Robin", robin_face)
    store_face(pet, "Claudia", claudia_face)
    arrived = pet.bus.subscribe(PersonArrived)
    left = pet.bus.subscribe(PersonLeft)
    pet.recognizer.engine.default = [sample(view(robin_face))]
    pet.face.show(face())
    assert (await asyncio.wait_for(arrived.get(), 2)).name == "Robin"

    pet.cfg.recognition.recheck_every_s = 0.05
    pet.recognizer.engine.default = [sample(view(claudia_face))]
    assert (await asyncio.wait_for(arrived.get(), 3)).name == "Claudia"
    assert (await asyncio.wait_for(left.get(), 2)).name == "Robin"
    assert pet.people.who_is_here() == (["Claudia"], 0)


async def test_a_recheck_keeps_only_the_crops_that_tell_something(pet, tmp_path):
    from PIL import Image
    from petd.vision.kept import FaceKeeper
    pet.recognizer.keeper = FaceKeeper(tmp_path, 200, lambda pid: {1: "Robin", 2: "Claudia"}.get(pid, "nobody"))
    robin_face, claudia_face = someone(), someone()
    store_face(pet, "Robin", robin_face)
    store_face(pet, "Claudia", claudia_face)

    def looks(face):
        s = sample(view(face, 0.3))
        s.crop = Image.new("RGB", (112, 112))
        return [s]
    kept = lambda: sorted(p.name for p in (tmp_path / "attempts").iterdir())
    arrived = pet.bus.subscribe(PersonArrived)
    pet.recognizer.engine.default = looks(robin_face)
    pet.face.show(face())
    assert (await asyncio.wait_for(arrived.get(), 2)).name == "Robin"
    for _ in range(100):                       # the first visit, with its confirmations
        if not pet.recognizer._visiting():
            break
        await asyncio.sleep(0.01)
    first_visit = kept()
    assert first_visit and all("as-Robin" in n for n in first_visit)

    pet.cfg.recognition.recheck_every_s = 0.05
    await asyncio.sleep(1.2)                   # rechecks that only confirm Robin
    assert kept() == first_visit

    pet.recognizer.engine.default = looks(claudia_face)
    assert (await asyncio.wait_for(arrived.get(), 3)).name == "Claudia"
    assert any("as-Claudia" in n for n in kept())


async def test_a_stranger_is_mentioned_after_a_few_clear_looks(pet):
    pet.cfg.memory.stranger_after_s = 0
    store_face(pet, "Robin", someone())
    pet.recognizer.engine.default = [sample(view(someone()))]
    pet.face.show(face())
    await asyncio.sleep(0.3)
    assert len(pet.told) == 1 and "don't recognize" in pet.told[0]
    assert pet.people.who_is_here() == ([], 1)


async def test_between_two_lookalikes_nobody_is_named(pet):
    robin_face = someone()
    twin = normalized(robin_face + 0.25 * someone())       # ~0.97 alike
    store_face(pet, "Robin", robin_face)
    store_face(pet, "Rob", twin)
    pet.recognizer.engine.default = [sample(normalized(robin_face + twin))]
    pet.face.show(face())
    await asyncio.sleep(0.3)
    assert pet.people.who_is_here() == ([], 1)          # the margin rule: not sure, so nobody


async def test_confident_looks_grow_the_fingerprints_but_not_duplicates(pet):
    robin_face = someone()
    robin = store_face(pet, "Robin", robin_face)
    arrived = pet.bus.subscribe(PersonArrived)
    pet.recognizer.engine.frames = [[sample(view(robin_face))] for _ in range(8)]
    pet.face.show(face())
    await asyncio.wait_for(arrived.get(), 2)
    await asyncio.sleep(0.3)
    grown = pet.db.face_count(robin.id) - 5
    assert grown == pet.cfg.recognition.grow_per_visit        # 8 new looks, 2 kept a visit

    # The same look over and over is one more fingerprint, not many.
    pet.face.show()
    await asyncio.sleep(0.3)
    same = sample(view(robin_face))
    pet.recognizer.engine.frames = []           # left over from the first visit
    pet.recognizer.engine.default = [same]
    before = pet.db.face_count(robin.id)
    pet.face.show(face())
    await asyncio.wait_for(arrived.get(), 2)
    await asyncio.sleep(0.3)
    assert pet.db.face_count(robin.id) - before <= 1


async def test_a_full_set_trades_its_most_redundant_grown_look_for_a_new_one(pet):
    pet.cfg.recognition.max_per_person = 6
    robin_face = someone()
    robin = store_face(pet, "Robin", robin_face)
    rows = pet.db.face_rows(robin.id)
    copy = normalized(np.frombuffer(rows[0][2], np.float32) + 0.05 * someone())   # ~0.999 alike
    pet.db.add_face_embedding(robin.id, to_blob(copy), source="grown")
    arrived = pet.bus.subscribe(PersonArrived)
    pet.recognizer.engine.frames = [[sample(view(robin_face))] for _ in range(4)]
    pet.face.show(face())
    await asyncio.wait_for(arrived.get(), 2)
    await asyncio.sleep(0.3)
    rows = pet.db.face_rows(robin.id)
    kept = [np.frombuffer(blob, np.float32) for _, _, blob in rows]
    assert len(rows) == 6                                          # full, but fresher:
    assert not any(np.allclose(v, copy.astype(np.float32)) for v in kept)   # the near-copy is gone
    assert [source for _, source, _ in rows].count("enroll") == 5  # enrolled ones stay


async def test_without_recognition_nobody_is_a_stranger(pet):
    await pet.recognizer.close()
    pet.recognizer = None
    pet.cfg.memory.stranger_after_s = 0
    pet.face.show(face(), face(cx=0.8))
    await asyncio.sleep(0.15)
    assert pet.told == []
    assert "sees 2 people]" in build_turn(pet, "hello")
    # The model is told, or it "recognizes" people from photos and memory.
    assert "never claim to recognise anyone" in build_system_prompt(pet)
    with pytest.raises(Exception, match="switched off"):
        await pet.people.enroll("Noah")


async def test_greetings_before_the_brain_is_up_are_kept():
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    app = App(cfg, fake=True)
    await app.start()
    try:
        robin = app.db.add_person("Robin")
        app.people.recognized(robin.id, time.time(), 0.8)
        told: list[str] = []
        app.people.set_notify(told.append)
        assert told and told[0].startswith("Robin")
    finally:
        await app.close()


# --- the prompt ----------------------------------------------------------------

async def test_prompt_lists_people_journal_and_facts(pet):
    robin = store_face(pet, "Robin", someone())
    pet.db.add_fact("hates Mondays", about=robin.id)
    pet.db.add_fact("the dock is behind the couch")
    pet.db.end_conversation(pet.db.start_conversation(), "Robin asked about the weather.")
    prompt = build_system_prompt(pet)
    assert "# People I know" in prompt and "- Robin: I know their face" in prompt
    assert "hates Mondays" in prompt and "new" not in prompt.split("# People I know")[1][:200]
    assert "Robin asked about the weather." in prompt
    assert "the dock is behind the couch" in prompt

    pet.people.recognized(robin.id, time.time(), 0.8)
    pet.face.show(face(), face(cx=0.8))
    await settle()
    turn = build_turn(pet, "hello", speaker="Robin")
    assert "sees Robin and someone I don't recognize" in turn
    assert 'Robin says: "hello"' in turn


def test_clean_summary_strips_tags():
    assert clean_summary(" [emote:happy] Met   Noah.\n[nod]") == "Met Noah."


# --- the brain's journal -------------------------------------------------------

class ScriptedBackend:
    def __init__(self):
        self.turns: list[str] = []
        self.episodes = 0
        self.ended = 0

    async def start_episode(self, system_prompt: str) -> None:
        self.episodes += 1

    async def send(self, user_turn: str):
        self.turns.append(user_turn)
        reply = "[emote:happy] Spoke with Noah about cake." if "[journal]" in user_turn else "Hello."
        yield TextDelta(reply)
        yield TurnDone()

    async def end_episode(self) -> None:
        self.ended += 1


async def test_idle_episode_writes_a_journal_entry(pet):
    pet.cfg.brain.episode_idle_timeout_s = 0.2
    pet.cfg.brain.prestart = False          # counts backend starts per episode
    backend = ScriptedBackend()
    brain = Brain(pet, backend)
    await brain.start()
    try:
        brain.tell("hi there", speaker="Noah")
        await asyncio.sleep(0.6)
        assert backend.episodes == 1 and backend.ended >= 1
        assert "[journal]" in backend.turns[-1]
        assert [c.summary for c in pet.db.journal()] == ["Spoke with Noah about cake."]
        conversation = pet.db.journal()[0].id
        assert pet.db.utterances(conversation) == [("Noah", "hi there"), ("pet", "Hello.")]

        # An episode of events alone (nobody spoke) leaves no journal entry.
        brain.tell("Robin just came into view.", kind="event")
        await asyncio.sleep(0.6)
        assert backend.episodes == 2 and len(pet.db.journal()) == 1
    finally:
        await brain.close()


async def test_the_backend_is_started_before_anyone_speaks(pet):
    pet.cfg.brain.episode_idle_timeout_s = 0.2
    backend = ScriptedBackend()
    brain = Brain(pet, backend)
    await brain.start()
    try:
        await asyncio.sleep(0.05)
        assert backend.episodes == 1 and backend.turns == []      # up, and waiting
        assert pet.db.journal() == []                             # no conversation yet
        brain.tell("hi there", speaker="Noah")
        await asyncio.sleep(0.1)
        assert backend.episodes == 1                              # the first turn used it
        await asyncio.sleep(0.5)                                  # idle: journal, then close
        assert backend.ended >= 1 and backend.episodes == 2       # the next one is up already
    finally:
        await brain.close()


def test_a_place_is_found_with_or_without_the():
    # 2026-09-28: the brain asked for "kitchen" first, every time, and was
    # told there was no such place: it's "the kitchen".
    db = MemoryDB(":memory:")
    db.set_place("the kitchen", 1.0, 2.0)
    db.set_place("Hallway", 3.0, 4.0)
    assert db.place("kitchen") == db.place("The  Kitchen") == (1.0, 2.0)
    assert db.place("the hallway") == (3.0, 4.0)
    assert db.place("the kitchenette") is None
    db.set_place("kitchen", 5.0, 6.0)             # the same place, moved: not a second one
    assert db.places() == ["Hallway", "the kitchen"] and db.place("the kitchen") == (5.0, 6.0)
