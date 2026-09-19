"""
The Brain: turns what the pet hears (and what happens around it) into
speech, expression and tool calls.

One "episode" is one LLM conversation. It starts on the first turn and
ends after `episode_idle_timeout_s` of quiet, so context stays small and
the persona reloads; cluster E adds a journal summary at that point.

Turns are serialized: while one is being spoken, another arrival waits
(or is dropped if the queue is already full), so the pet never talks over
itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from ..events import Heard, SpeechEnded, SpeechStarted
from . import prompt as prompt_module
from .backend import BrainError, LLMBackend, TextDelta, ToolFinished, ToolStarted, TurnDone
from .expressions import Expressions
from .tags import Action, Sentence, SpeechStreamParser

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)


class Brain:
    def __init__(self, pet: "App", backend: LLMBackend):
        self.pet = pet
        self.backend = backend
        self.cfg = pet.cfg.brain
        self.expressions: Optional[Expressions] = (
            Expressions(pet.face) if pet.face is not None else None)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._tasks: list[asyncio.Task] = []
        self._last_turn_at = 0.0
        self._episode_open = False
        self.busy = False

    async def start(self) -> None:
        subs = [self.pet.bus.subscribe(Heard),
                self.pet.bus.subscribe(SpeechStarted, SpeechEnded)]
        self._tasks = [
            asyncio.create_task(self._collect_heard(subs[0]), name="brain-heard"),
            asyncio.create_task(self._feedback_loop(subs[1]), name="brain-feedback"),
            asyncio.create_task(self._run_turns(), name="brain-turns"),
        ]

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.backend.end_episode()

    # --- inputs ---------------------------------------------------------------

    async def _collect_heard(self, sub) -> None:
        async for event in sub:
            self.tell(event.text, kind="heard")

    def tell(self, text: str, kind: str = "heard", speaker: Optional[str] = None) -> None:
        """Queues a turn. Behaviors use kind='event' to report what happened."""
        try:
            self._queue.put_nowait((text, kind, speaker))
        except asyncio.QueueFull:
            log.warning("brain is behind; dropping %s: %r", kind, text)

    async def _feedback_loop(self, sub) -> None:
        """Immediate eye feedback so the ~2s think time doesn't feel dead."""
        if self.expressions is None:
            return
        async for event in sub:
            try:
                if isinstance(event, SpeechStarted):
                    await self.expressions.listening()
                elif isinstance(event, SpeechEnded) and not event.discarded:
                    await self.expressions.thinking()
            except Exception:  # noqa: BLE001
                log.debug("feedback expression failed", exc_info=True)

    # --- the turn loop --------------------------------------------------------

    async def _run_turns(self) -> None:
        while True:
            text, kind, speaker = await self._queue.get()
            try:
                await self._run_turn(text, kind, speaker)
            except Exception:  # noqa: BLE001 - one bad turn must not kill the brain
                log.exception("turn failed")

    async def _ensure_episode(self) -> None:
        idle = time.time() - self._last_turn_at
        if self._episode_open and idle < self.cfg.episode_idle_timeout_s:
            return
        if self._episode_open:
            log.info("episode idle for %.0fs; starting a fresh one", idle)
            await self.backend.end_episode()
        await self.backend.start_episode(prompt_module.build_system_prompt(self.pet))
        self._episode_open = True

    async def _run_turn(self, text: str, kind: str, speaker: Optional[str]) -> None:
        await self._ensure_episode()
        turn = prompt_module.build_turn(self.pet, text, kind=kind, speaker=speaker)
        log.info("-> brain: %s", turn.replace("\n", " | "))

        parser = SpeechStreamParser()
        utterance = None
        self.busy = True
        started = time.monotonic()
        try:
            async for event in self.backend.send(turn):
                if isinstance(event, TextDelta):
                    log.debug("delta %r", event.text)
                    for piece in parser.feed(event.text):
                        utterance = await self._emit(piece, utterance)
                elif isinstance(event, ToolStarted):
                    log.info("brain calls %s(%s)", event.name, event.arguments)
                elif isinstance(event, ToolFinished):
                    if event.is_error:
                        log.info("tool reported an error back to the brain")
                elif isinstance(event, BrainError):
                    log.error("brain error: %s", event.message)
                elif isinstance(event, TurnDone):
                    for piece in parser.flush():
                        utterance = await self._emit(piece, utterance)
                    log.info("turn done in %.1fs (cost %s)",
                             event.duration_s or (time.monotonic() - started), event.cost_usd)
        finally:
            self.busy = False
            self._last_turn_at = time.time()
            if utterance is not None:
                utterance.end()
                await utterance.wait()
            if self.expressions is not None:
                await self.expressions.rest()

    async def _emit(self, piece, utterance):
        if isinstance(piece, Action):
            if self.expressions is not None:
                await self.expressions.apply(piece)
            return utterance
        if isinstance(piece, Sentence):
            log.info("<- brain says: %s", piece.text)
            if self.pet.speaker is not None:
                if utterance is None:
                    utterance = self.pet.speaker.begin()
                utterance.add(piece.text)
        return utterance


def build_brain(pet: "App") -> Optional[Brain]:
    cfg = pet.cfg.brain
    if not cfg.enabled:
        return None
    if cfg.backend == "ollama":
        from .ollama import OllamaBackend
        backend: LLMBackend = OllamaBackend(pet.cfg, pet.tools)
    else:
        from .claude_cli import ClaudeCliBackend
        backend = ClaudeCliBackend(pet.cfg)
    return Brain(pet, backend)
