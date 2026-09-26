"""
What the pet does with what it hears (PLAN.md 4.3, E2): reflexes first,
then the attention gate, then the brain.

- Reflexes are handled here, without the LLM, and the brain only hears
  about them afterwards. "Stop" and "be quiet" are always live, even
  mid-sentence: the echo gate drops everything heard while the pet talks,
  so those transcripts are checked for a reflex word the pet didn't just
  say itself (a barge-in). "Go home" and "go to sleep" need the pet's
  attention, since they move it.
- The gate lets speech through to the brain when the pet's name is in it,
  when a conversation window is open (it spoke, or was addressed, in the
  last `window_s`), or when a known person is in view (any face, while
  recognition is off). Everything else is
  published as HeardDropped("not addressed") and left alone.
- Echo is filtered twice: by time in io/stt.py (while the pet talks, plus
  a tail), and here by content, for echoes that outlast that estimate:
  a transcript that mostly repeats the pet's last 20 s of speech is not
  someone talking to it.
- The strict reflex match (the whole utterance is the command, give or take
  the name and a "please") keeps "don't stop" or "stop by the shop later"
  from halting the robot.
"""

from __future__ import annotations

import asyncio
import logging
import re
import string
import time
from typing import TYPE_CHECKING, Optional

from ..events import Heard, HeardDropped, SpeakingFinished
from ..io.stt import normalize

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

REFLEXES: dict[str, tuple[str, ...]] = {
    "stop": ("stop", "halt", "freeze", "stop it", "stop that", "stop moving", "stand still",
             "stop stop", "stop right there"),
    "quiet": ("quiet", "be quiet", "shut up", "silence", "hush", "shh", "shush", "enough",
              "thats enough", "stop talking", "quiet please"),
    "home": ("go home", "go to your dock", "go to the dock", "go back to your dock",
             "go back to the dock", "go dock", "dock", "back to your dock"),
    "sleep": ("go to sleep", "sleep", "time to sleep", "go to bed"),
}
# Reflexes that don't need the pet's attention: harmless, and urgent.
ALWAYS = ("stop", "quiet")
# Words that may appear in a transcript of someone talking over the pet.
BARGE_IN = {"stop": "stop", "halt": "stop", "freeze": "stop",
            "quiet": "quiet", "shut": "quiet", "silence": "quiet", "enough": "quiet"}
# How far back our own words count when recognizing an echo by its text.
ECHO_WINDOW_S = 10.0 # 20.0
# An echo starts while our voice is audible or just after; a transcript that
# starts later than this after it is someone talking, even in our words
# ("Please get Claudia" to "should I get Claudia?").
ECHO_START_S = 3.0
FILLERS = {"please", "now", "right", "ok", "okay", "hey", "just", "oh", "come", "on", "you",
           "yes", "no", "i", "said"}


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


class NameMatcher:
    """
    Finds the pet's name in a transcript, however whisper spelled it.

    Whisper is primed with the name (stt.prompt), and its near misses then
    keep the name's capitals: "GularDOS", "OkGLaDOS". A word ending in
    the name's last three capitals, or containing the name, counts.
    """

    def __init__(self, name: str, wake_words: list[str]):
        self.name = normalize(name).replace(" ", "")
        self.tail = re.sub(r"[^A-Za-z]", "", name)[-3:] if name[-3:].isupper() else None
        self.phrases = {normalize(w) for w in wake_words} | {self.name}

    def split(self, text: str) -> tuple[bool, list[str]]:
        """(was the name said, the remaining words)."""
        raw = [w.strip(string.punctuation) for w in text.split()]
        raw = [w for w in raw if w]
        words = [normalize(w) for w in raw]
        rest: list[str] = []
        found = False
        i = 0
        while i < len(words):
            pair = " ".join(words[i:i + 2])
            if len(words) > i + 1 and (pair in self.phrases or self._close(words[i] + words[i + 1])):
                found = True
                i += 2
            elif words[i] in self.phrases or self._close(words[i]) or self._marked(raw[i]):
                found = True
                i += 1
            else:
                rest.extend(words[i].split())
                i += 1
        return found, rest

    def _close(self, word: str) -> bool:
        # One edit only: two lets in "gladly" and "glass".
        return len(word) >= 5 and edit_distance(word, self.name) <= 1

    def _marked(self, raw: str) -> bool:
        if self.name in raw.lower().replace("'", ""):
            return True
        return (self.tail is not None and len(raw) > len(self.tail)
                and raw.endswith(self.tail))


def match_reflex(words: list[str]) -> Optional[str]:
    """The reflex this utterance *is* (not merely mentions), if any."""
    core = " ".join(w for w in words if w not in FILLERS).replace("'", "")
    for reflex, phrases in REFLEXES.items():
        if core in phrases:
            return reflex
    return None


def barge_in(heard: str, said: str) -> Optional[str]:
    """A stop/quiet word in an echo-gated transcript that wasn't in our own speech."""
    ours = set(normalize(said).split())
    for word in normalize(heard).split():
        if word in BARGE_IN and word not in ours:
            return BARGE_IN[word]
    return None


def repeats_own_speech(heard: str, said: str) -> bool:
    """
    Mostly our own recent words: an echo the timing gate let through
    (it ends as soon as we think the robot's speaker went quiet).
    """
    words = normalize(heard).split()
    ours = set(normalize(said).split())
    if not words or not ours:
        return False
    matched = sum(word in ours for word in words)
    if len(words) == 1:
        # One word is only an echo if it's how our last sentence ended.
        return normalize(said).split()[-1] == words[0]
    return matched >= 2 and matched / len(words) >= 0.6


class Listener:
    def __init__(self, pet: "App"):
        self.pet = pet
        self.cfg = pet.cfg.converse
        self.names = NameMatcher(pet.cfg.pet.name, self.cfg.wake_words)
        self.window_until = 0.0
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        sub = self.pet.bus.subscribe(Heard, HeardDropped, SpeakingFinished)
        self._task = asyncio.create_task(self._run(sub), name="listener")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()

    def window_open(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) < self.window_until

    async def _run(self, sub) -> None:
        async for event in sub:
            try:
                if isinstance(event, SpeakingFinished):
                    if not event.interrupted:
                        self.window_until = event.t + self.cfg.window_s
                elif isinstance(event, HeardDropped):
                    if event.reason.startswith("echo"):
                        await self._maybe_barge_in(event.text)
                else:
                    await self.on_heard(event)
            except Exception:  # noqa: BLE001 - keep listening whatever happens
                log.exception("listener failed on %r", event)

    async def _maybe_barge_in(self, text: str) -> None:
        speaker = self.pet.speaker
        reflex = barge_in(text, speaker.said_recently() if speaker else "")
        if reflex:
            log.warning("barge-in %r -> %s", text, reflex)
            await self.reflex(reflex, text)

    async def on_heard(self, event: Heard) -> None:
        speaker = self.pet.speaker
        if (speaker is not None and speaker.overlaps(event.t_start - ECHO_START_S, event.t_end)
                and repeats_own_speech(event.text, speaker.said_recently(ECHO_WINDOW_S))):
            log.info("ignoring %r: repeats what I just said", event.text)
            self.pet.bus.publish(HeardDropped(text=event.text, reason="echo (repeats own words)"))
            return   # HeardDropped("echo...") still gets its barge-in check

        addressed, words = self.names.split(event.text)
        reflex = match_reflex(words)
        if reflex in ALWAYS:
            await self.reflex(reflex, event.text)
            return

        if self.cfg.ignore_while_driving and self._driving() and not addressed:
            return self._ignore(event.text, "driving")
        if not (addressed or self.window_open(event.t) or self._gaze()):
            return self._ignore(event.text, "not addressed")
        self.window_until = event.t + self.cfg.window_s

        if reflex is not None:
            await self.reflex(reflex, event.text)
            return
        brain = self.pet.brain
        if brain is not None:
            person = self.pet.people.sole_person() if self.pet.people else None
            brain.tell(event.text, kind="heard", speaker=person.name if person else None)

    def _driving(self) -> bool:
        """
        Wheels turning: a turn or move under way, or Valetudo driving by
        itself. Not manual control left armed between motions (the lidar keeps
        spinning for idle_disarm_s): Valetudo reports that as manual_control
        while the robot stands still, and 20 s after every move went unheard.
        """
        vacuum = self.pet.vacuum
        return self.pet.moving or (vacuum is not None
                                   and vacuum.state.status in ("moving", "returning", "cleaning"))

    def _gaze(self) -> bool:
        if not self.cfg.gaze_opens or self.pet.face is None or not self.pet.face.presence.present:
            return False
        if not self.pet.recognition_on:
            return True     # nobody can be known without it: a face in view will do
        return self.pet.people is not None and bool(self.pet.people.present)

    def _ignore(self, text: str, reason: str) -> None:
        log.info("ignoring %r: %s", text, reason)
        self.pet.bus.publish(HeardDropped(text=text, reason=reason))
        brain = self.pet.brain
        if brain is not None and brain.expressions is not None and not brain.busy:
            # SpeechEnded already set the eye to "thinking"; nothing to think about.
            asyncio.create_task(brain.expressions.rest())

    async def reflex(self, reflex: str, text: str) -> None:
        log.warning("reflex %s (heard %r)", reflex, text)
        pet, brain = self.pet, self.pet.brain
        if reflex == "stop":
            await pet.stop_everything("voice")
            if brain is not None:
                brain.note(f'Someone said "{text}", so you stopped moving and talking at once.')
        elif reflex == "quiet":
            if brain is not None:
                brain.hush()
            if pet.speaker is not None:
                pet.speaker.interrupt()
            if brain is not None:
                brain.note(f'Someone said "{text}", so you stopped talking mid-sentence.')
        elif reflex in ("home", "sleep"):
            if pet.vacuum is None or pet.dock is None:
                return
            await pet.go_home()
            if brain is not None:
                brain.tell(f'Someone said "{text}". You are already driving back to your dock.',
                           kind="event")
