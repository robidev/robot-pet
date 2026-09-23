import asyncio
import time

import pytest

from petd.app import App
from petd.brain.backend import TextDelta, TurnDone
from petd.brain.brain import Brain
from petd.brain.prompt import build_system_prompt, build_turn, clean_summary
from petd.config import Config
from petd.events import PersonArrived, PersonLeft
from petd.io.face import Face
from petd.memory.db import MemoryDB


def face(id=-1, cx=0.5):
    return Face(id, 0.9, cx - 0.1, 0.3, cx + 0.1, 0.6)


@pytest.fixture
async def pet():
    cfg = Config()
    cfg.api.enabled = False
    cfg.brain.enabled = False
    cfg.stt.enabled = False
    cfg.speaker.enabled = False
    cfg.face.faces_lost_debounce_s = 0.1
    cfg.memory.enroll_timeout_s = 0.5
    # These test recognition itself, which is off by default for now.
    cfg.face.enable_recognition = True
    app = App(cfg, fake=True)
    await app.start()
    told: list[str] = []
    app.people.set_notify(told.append)
    app.people.poll_s = 0
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

async def test_enroll_stores_the_new_slot_and_skips_the_greeting(pet):
    pet.face.show(face())
    result = await pet.people.enroll("Noah")
    assert "Noah" in result
    noah = pet.db.person_by_name("Noah")
    assert noah.face_slot == 0 and pet.face.enrolled == [0]
    assert pet.people.who_is_here() == (["Noah"], 0)

    # The device now recognizes them: no greeting, they're mid-conversation.
    pet.face.show(face(id=0))
    await settle()
    assert pet.told == []


async def test_enroll_refuses_nobody_crowds_and_lookalikes(pet):
    with pytest.raises(Exception, match="can't see a face"):
        await pet.people.enroll("Noah")

    pet.face.show(face(cx=0.3), face(cx=0.7))
    with pytest.raises(Exception, match="more than one face"):
        await pet.people.enroll("Noah")

    pet.face.enrolled = [4]
    pet.db.add_person("Robin", face_slot=4)
    pet.face.show(face(id=4))
    with pytest.raises(Exception, match="looks like Robin"):
        await pet.people.enroll("Noah")
    assert pet.db.person_by_name("Noah") is None

    # ...unless the model insists, after the person does.
    await pet.people.enroll("Noah", insist=True)
    assert pet.db.person_by_name("Noah").face_slot == 5


async def test_enroll_adopts_a_face_stored_without_a_name(pet):
    pet.face.enrolled = [1]
    pet.face.show(face(id=1))
    result = await pet.people.enroll("Robin")
    assert "already had this face" in result
    assert pet.db.person_by_name("Robin").face_slot == 1
    assert ("enroll",) not in pet.face.commands


async def test_enroll_when_full_and_when_nothing_happens(pet):
    pet.face.enrolled = list(range(7))
    pet.face.show(face())
    with pytest.raises(Exception, match="full"):
        await pet.people.enroll("Noah")

    pet.face.enrolled = []
    real_enroll = pet.face.enroll_next_face

    async def never_enrolls():
        pet.face.commands.append(("enroll",))
    pet.face.enroll_next_face = never_enrolls
    with pytest.raises(Exception, match="didn't manage"):
        await pet.people.enroll("Noah")
    assert pet.face.commands[-1] == ("enroll_cancel",)
    pet.face.enroll_next_face = real_enroll


async def test_forget_deletes_slot_and_row(pet):
    pet.face.show(face())
    await pet.people.enroll("Noah")
    result = await pet.people.forget("noah")
    assert "Noah" in result
    assert pet.face.enrolled == [] and pet.db.person_by_name("Noah") is None
    with pytest.raises(Exception, match="don't know anyone"):
        await pet.people.forget("Noah")


async def test_reconcile_unlinks_vanished_slots(pet):
    pet.db.add_person("Robin", face_slot=2)
    pet.face.enrolled = [5]
    assert await pet.people.reconcile() == [5]
    assert pet.db.person_by_name("Robin").face_slot is None


# --- sticky identity and greetings ---------------------------------------------

async def test_identity_sticks_through_flicker_and_greets_once(pet):
    robin = pet.db.add_person("Robin", face_slot=1)
    arrived = pet.bus.subscribe(PersonArrived)
    left = pet.bus.subscribe(PersonLeft)

    pet.face.show(face(id=1))
    assert (await asyncio.wait_for(arrived.get(), 1)).name == "Robin"
    await settle()
    assert len(pet.told) == 1 and pet.told[0].startswith("Robin just came into view")

    pet.face.show(face(id=-1))              # recognition flickers...
    await settle()
    assert pet.people.who_is_here() == (["Robin"], 0)
    assert pet.people.sole_person().id == robin.id

    pet.face.show(face(id=-1), face(id=-1, cx=0.8))
    await settle()
    assert pet.people.who_is_here() == (["Robin"], 1)
    assert pet.people.sole_person() is None

    pet.face.show()                         # ...and only nobody-in-view ends it
    assert (await asyncio.wait_for(left.get(), 1)).name == "Robin"
    assert pet.people.who_is_here() == ([], 0)

    pet.face.show(face(id=1))               # back within greet_every_h: no second greeting
    await asyncio.wait_for(arrived.get(), 1)
    await settle()
    assert len(pet.told) == 1


async def test_a_lingering_stranger_is_mentioned(pet):
    pet.cfg.memory.stranger_after_s = 0.05
    pet.face.show(face())
    await asyncio.sleep(0.15)
    assert len(pet.told) == 1 and "don't recognize" in pet.told[0]


async def test_without_recognition_nobody_is_a_stranger(pet):
    pet.cfg.face.enable_recognition = False
    pet.cfg.memory.stranger_after_s = 0.05
    pet.face.show(face(), face(cx=0.8))
    await asyncio.sleep(0.15)
    assert pet.told == []
    assert "sees 2 people]" in build_turn(pet, "hello")
    with pytest.raises(Exception, match="switched off"):
        await pet.people.enroll("Noah")
    assert pet.face.enrolled == []


async def test_greetings_before_the_brain_is_up_are_kept():
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.face.enable_recognition = True
    app = App(cfg, fake=True)
    await app.start()
    try:
        app.db.add_person("Robin", face_slot=0)
        app.face.show(face(id=0))
        await settle()
        told: list[str] = []
        app.people.set_notify(told.append)
        assert told and told[0].startswith("Robin")
    finally:
        await app.close()


# --- the prompt ----------------------------------------------------------------

async def test_prompt_lists_people_journal_and_facts(pet):
    robin = pet.db.add_person("Robin", face_slot=1)
    pet.db.add_fact("hates Mondays", about=robin.id)
    pet.db.add_fact("the dock is behind the couch")
    pet.db.end_conversation(pet.db.start_conversation(), "Robin asked about the weather.")
    prompt = build_system_prompt(pet)
    assert "# People I know" in prompt and "- Robin: I know their face" in prompt
    assert "hates Mondays" in prompt and "new" not in prompt.split("# People I know")[1][:200]
    assert "Robin asked about the weather." in prompt
    assert "the dock is behind the couch" in prompt

    pet.face.show(face(id=1), face(id=-1, cx=0.8))
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
