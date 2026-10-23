import asyncio
import base64

import pytest

from petd.app import App
from petd.brain.backend import TextDelta, TurnDone
from petd.brain.brain import Brain
from petd.brain.claude_cli import user_message
from petd.config import Config
from petd.events import SpeechStarted
from petd.io.face import Face


class RecordingBackend:
    def __init__(self):
        self.turns: list[tuple[str, list]] = []

    async def start_episode(self, system_prompt: str) -> None: ...
    async def end_episode(self) -> None: ...

    async def send(self, user_turn: str, images=()):
        self.turns.append((user_turn, list(images)))
        yield TextDelta("Hm.")
        yield TurnDone()


@pytest.fixture
async def pet():
    cfg = Config()
    cfg.api.enabled = cfg.brain.enabled = cfg.stt.enabled = cfg.speaker.enabled = False
    cfg.recognition.enabled = False         # its own snapshots would muddle the counts
    cfg.face.faces_lost_debounce_s = 0.1
    cfg.brain.prestart = False
    cfg.brain.photo_every_s = 120.0
    app = App(cfg, fake=True)
    await app.start()
    app.backend = RecordingBackend()
    app.brain = Brain(app, app.backend)
    await app.brain.start()
    try:
        yield app
    finally:
        await app.brain.close()
        await app.close()


async def turn(pet, text, kind="heard"):
    before = len(pet.backend.turns)
    pet.brain.tell(text, kind=kind, speaker="Robin")
    for _ in range(200):
        if len(pet.backend.turns) > before:
            return pet.backend.turns[-1]
        await asyncio.sleep(0.01)
    raise AssertionError("no turn")


def snapshots(pet):
    return sum(c == ("snapshot",) for c in pet.face.commands)


async def someone_in_view(pet):
    pet.face.show(Face(-1, 0.9, 0.4, 0.3, 0.6, 0.6))
    await asyncio.sleep(0.02)


async def test_the_first_words_in_a_conversation_come_with_a_photo_then_now_and_then(pet):
    # "Can you see the gesture I'm making?" got "I can't see your hands",
    # with no photo taken (2026-10-23).
    await someone_in_view(pet)
    text, images = await turn(pet, "Can you see the gesture I'm making?")
    assert len(images) == 1 and images[0] == pet.face.last_snapshot[1].jpeg
    assert "[my camera, as they spoke, looking " in text
    assert text.index("[my camera") < text.index('Robin says: "Can you see')

    text, images = await turn(pet, "And now?")
    assert images == [] and "[my camera" not in text            # not again so soon
    _, images = await turn(pet, "Robin moved.", kind="event")
    assert images == []                                         # only with someone talking

    pet.brain.photos.every_s = 0.05
    await asyncio.sleep(0.06)
    _, images = await turn(pet, "And now?")
    assert len(images) == 1


async def test_no_photo_with_nobody_in_view_or_with_it_switched_off(pet):
    _, images = await turn(pet, "Hello?")
    assert images == [] and snapshots(pet) == 0

    pet.brain.photos.every_s = 0
    await someone_in_view(pet)
    _, images = await turn(pet, "Hello?")
    assert images == [] and snapshots(pet) == 0


async def test_the_photo_is_taken_while_they_are_still_talking(pet):
    await someone_in_view(pet)
    pet.bus.publish(SpeechStarted(t_utc=0.0))
    for _ in range(100):
        if snapshots(pet):
            break
        await asyncio.sleep(0.01)
    assert snapshots(pet) == 1
    _, images = await turn(pet, "How many fingers am I holding up?")
    assert len(images) == 1 and snapshots(pet) == 1             # that one, not a second


async def test_a_new_conversation_gets_a_photo_again(pet):
    await someone_in_view(pet)
    _, images = await turn(pet, "Hello.")
    assert len(images) == 1
    pet.brain.photos.new_conversation()
    _, images = await turn(pet, "Hello again.")
    assert len(images) == 1


def test_the_cli_turn_carries_the_photo_before_the_text():
    message = user_message("[21:14 | sees Robin] Robin says: \"hi\"", [b"\xff\xd8jpeg"])
    image, text = message["message"]["content"]
    assert image == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.b64encode(b"\xff\xd8jpeg").decode()}}
    assert text["type"] == "text" and "Robin says" in text["text"]
    assert user_message("hi")["message"]["content"] == [{"type": "text", "text": "hi"}]
