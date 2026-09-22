import asyncio
import time

import pytest

from petd.app import App
from petd.behavior.converse import Listener, NameMatcher, barge_in, match_reflex
from petd.config import Config
from petd.events import Heard, HeardDropped, SpeakingFinished
from petd.io.face import Face


def words_after_name(text):
    return NameMatcher("GLaDOS", Config().converse.wake_words).split(text)


@pytest.mark.parametrize("text", [
    "GLaDOS, what time is it?", "Gladys what time is it", "hey glad os, what time is it",
    "Gladis? What time is it", "what time is it, Glados", "Gladdos, what time is it"])
def test_the_name_however_whisper_spells_it(text):
    found, rest = words_after_name(text)
    assert found
    assert [w for w in rest if w != "hey"] == ["what", "time", "is", "it"]


@pytest.mark.parametrize("text", [
    "I'd gladly do it", "a glass of water", "the gladiolus", "what time is it"])
def test_words_that_are_not_the_name(text):
    assert not words_after_name(text)[0]


@pytest.mark.parametrize("text,reflex", [
    ("Stop!", "stop"), ("please stop", "stop"), ("GLaDOS, stop it.", "stop"),
    ("Freeze!", "stop"), ("Be quiet.", "quiet"), ("That's enough!", "quiet"),
    ("Shut up, Gladys", "quiet"), ("Gladys, go home", "home"), ("Go back to your dock.", "home"),
    ("GLaDOS go to sleep", "sleep"),
    ("don't stop", None), ("stop by the shop later", None), ("I need to go home soon", None),
    ("is it quiet in here", None), ("what time is it", None)])
def test_reflexes_match_whole_commands_only(text, reflex):
    assert match_reflex(words_after_name(text)[1]) == reflex


def test_barge_in_ignores_our_own_words():
    assert barge_in("the cake is a stop lie", "The cake is a lie.") == "stop"
    assert barge_in("stop worrying about it", "Stop worrying about it.") is None
    assert barge_in("shut the", "") == "quiet"
    assert barge_in("the cake is a lie", "The cake is a lie.") is None


class StubBrain:
    def __init__(self):
        self.told, self.notes, self.hushed = [], [], 0
        self.expressions = None
        self.busy = False

    def tell(self, text, kind="heard", speaker=None):
        self.told.append((text, kind, speaker))

    def note(self, text):
        self.notes.append(text)

    def hush(self):
        self.hushed += 1

    async def close(self):
        pass


@pytest.fixture
async def pet():
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = False
    cfg.speaker.playback_latency_s = 0.0
    cfg.face.faces_lost_debounce_s = 0.1
    app = App(cfg, fake=True)
    await app.start()
    app.brain = StubBrain()
    app.listener = Listener(app)
    await app.listener.start()
    dropped = app.bus.subscribe(HeardDropped)
    app.dropped = dropped
    try:
        yield app
    finally:
        await app.listener.close()
        await app.close()


async def hear(pet, text):
    pet.bus.publish(Heard(text=text, t_start=time.time() - 1, t_end=time.time()))
    for _ in range(5):
        await asyncio.sleep(0)


async def test_only_speech_meant_for_the_pet_reaches_the_brain(pet):
    await hear(pet, "did you feed the cat")
    assert pet.brain.told == []
    assert (await asyncio.wait_for(pet.dropped.get(), 1)).reason == "not addressed"

    await hear(pet, "Gladys, did you feed the cat?")
    assert pet.brain.told[-1][0] == "Gladys, did you feed the cat?"

    # Addressed once, the conversation stays open for a while...
    await hear(pet, "and the fish?")
    assert pet.brain.told[-1][0] == "and the fish?"

    # ...and closes again.
    pet.listener.window_until = 0
    await hear(pet, "never mind")
    assert len(pet.brain.told) == 2


async def test_the_pet_speaking_opens_the_window(pet):
    pet.bus.publish(SpeakingFinished(utterance_id=1, text="Oh. It's you."))
    await hear(pet, "yes it's me")
    assert pet.brain.told[-1][0] == "yes it's me"


async def test_a_known_face_in_view_opens_the_gate(pet):
    pet.db.add_person("Robin", face_slot=0)
    pet.face.show(Face(0, 0.9, 0.4, 0.3, 0.6, 0.7))
    await asyncio.sleep(0.05)
    await hear(pet, "what do you think about that")
    assert pet.brain.told[-1] == ("what do you think about that", "heard", "Robin")


async def test_stop_works_unaddressed_and_mid_sentence(pet):
    await hear(pet, "Stop!")
    assert ("stop",) in pet.vacuum.commands
    assert pet.brain.told == [] and "stopped moving" in pet.brain.notes[-1]

    # While speaking, the mic transcript is dropped as echo; a stop word we
    # didn't say ourselves still counts.
    pet.vacuum.commands.clear()
    pet.speaker.say("I have been thinking about the nature of cake.")
    await asyncio.sleep(0.2)
    pet.stt.inject("the nature of stop cake")
    await asyncio.sleep(0.05)
    assert ("stop",) in pet.vacuum.commands


async def test_be_quiet_hushes_without_stopping_the_wheels(pet):
    await hear(pet, "be quiet")
    assert pet.brain.hushed == 1 and ("stop",) not in pet.vacuum.commands


async def test_go_home_needs_attention(pet):
    await hear(pet, "go home")
    assert not any(c[0] == "dock" for c in pet.vacuum.commands)
    await hear(pet, "Gladys, go home")
    assert any(c[0] == "dock" for c in pet.vacuum.commands)
    assert pet.brain.told[-1][1] == "event"


async def test_driving_noise_is_ignored_unless_named(pet):
    pet.vacuum._update(status="moving")      # the fake arrives instantly; hold it mid-drive
    assert pet.vacuum.state.moving
    pet.listener.window_until = time.time() + 60
    await hear(pet, "brrrm rattle")
    assert pet.brain.told == []
    await hear(pet, "Gladys, where are you going?")
    assert pet.brain.told[-1][0] == "Gladys, where are you going?"
