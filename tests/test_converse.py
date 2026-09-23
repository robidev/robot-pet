import asyncio
import time

import pytest

from petd.app import App
from petd.behavior import converse
from petd.behavior.converse import (Listener, NameMatcher, barge_in, match_reflex,
                                    repeats_own_speech)
from petd.config import Config
from petd.events import Heard, HeardDropped, SpeakingFinished
from petd.io.face import Face


def words_after_name(text):
    return NameMatcher("GLaDOS", Config().converse.wake_words).split(text)


@pytest.mark.parametrize("text", [
    "GLaDOS, what time is it?", "Gladys what time is it", "hey glad os, what time is it",
    "Gladis? What time is it", "what time is it, Glados", "Gladdos, what time is it",
    # what whisper makes of the name once primed with it
    "GularDOS, what time is it?", "OkGLaDOS, what time is it", "JledDOS, what time is it"])
def test_the_name_however_whisper_spells_it(text):
    found, rest = words_after_name(text)
    assert found
    assert [w for w in rest if w != "hey"] == ["what", "time", "is", "it"]


@pytest.mark.parametrize("text", [
    "I'd gladly do it", "a glass of water", "the gladiolus", "what time is it",
    "the DOS prompt", "those kudos"])
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


@pytest.mark.parametrize("heard,echo", [
    ("the cake is a lie", True), ("is a lie", True), ("cake is a lie Robin", True),
    ("Lie.", True), ("Cake.", False), ("why do you say that I'm new", False),
    ("what cake", False), ("", False)])
def test_echo_by_content(heard, echo):
    assert repeats_own_speech(heard, "Hello. You're new. The cake is a lie.") is echo


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


async def until_audible(pet):
    while not pet.speaker.speaking:
        await asyncio.sleep(0.01)


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
    robin = pet.db.add_person("Robin")
    pet.face.show(Face(-1, 0.9, 0.4, 0.3, 0.6, 0.7))
    pet.people.recognized(robin.id, time.time(), 0.8)
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
    await until_audible(pet)
    await asyncio.sleep(1.1)        # the injected second is all inside our voice
    pet.stt.inject("the nature of stop cake")
    await asyncio.sleep(0.05)
    assert ("stop",) in pet.vacuum.commands


async def test_be_quiet_hushes_without_stopping_the_wheels(pet):
    await hear(pet, "be quiet")
    assert pet.brain.hushed == 1 and ("stop",) not in pet.vacuum.commands


async def test_go_home_needs_attention(pet):
    pet.vacuum._update(status="idle")
    pet.dock.poll_s = pet.dock.start_grace_s = 0.02
    await hear(pet, "go home")
    assert not any(c[0] == "dock" for c in pet.vacuum.commands)
    await hear(pet, "Gladys, go home")
    await asyncio.wait_for(pet.motion_task, 5)
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


async def test_a_late_echo_is_not_a_conversation(pet):
    pet.speaker.say("Consider yourself officially remembered.")
    await until_audible(pet)
    pet.listener.window_until = time.time() + 60
    await hear(pet, "officially remembered")
    assert pet.brain.told == []
    assert (await asyncio.wait_for(pet.dropped.get(), 1)).reason.startswith("echo")


async def test_quoting_the_pet_later_is_not_an_echo(pet):
    # 11:28 on 2026-09-23: "Please get Claudia", 12 s after "...or should I get
    # Claudia?", was dropped for repeating the pet's words.
    pet.cfg.speaker.gate_tail_s = 0.0
    pet.speaker.say("Do you need to sit down, or should I get Claudia?")
    await until_audible(pet)
    while pet.speaker.speaking:
        await asyncio.sleep(0.01)
    await asyncio.sleep(converse.ECHO_START_S + 1.2)   # hear() starts the utterance 1 s back
    pet.listener.window_until = time.time() + 60
    await hear(pet, "Please get Claudia")
    assert pet.brain.told[-1][0] == "Please get Claudia"


async def test_any_face_opens_the_gate_without_recognition(pet):
    await pet.recognizer.close()
    pet.recognizer = None
    pet.face.show(Face(-1, 0.9, 0.4, 0.3, 0.6, 0.6))
    await asyncio.sleep(0.05)
    await hear(pet, "can you hear me?")
    assert pet.brain.told[-1][0] == "can you hear me?"


async def test_the_lidar_left_on_between_moves_is_not_driving(pet):
    # 2026-09-23 13:28: manual control stays armed for 20 s after a move, and
    # Valetudo calls that manual_control; the robot stands still meanwhile.
    pet.vacuum._update(status="manual_control")
    await hear(pet, "drive one more meter")
    assert pet.brain.told == []                  # not addressed: nothing open, nobody in view
    assert (await asyncio.wait_for(pet.dropped.get(), 1)).reason == "not addressed"
    pet.vacuum._update(status="returning")       # Valetudo driving by itself
    await hear(pet, "thank god")
    assert (await asyncio.wait_for(pet.dropped.get(), 1)).reason == "driving"
