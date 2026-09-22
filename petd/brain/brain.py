"""
The Brain: turns what the pet hears (and what happens around it) into
speech, expression and tool calls.

One "episode" is one LLM conversation. It starts on the first turn and
ends after `episode_idle_timeout_s` of quiet, so context stays small and
the persona reloads. Before it ends, the model writes a line for its
journal (never spoken), and the next episode's system prompt carries the
most recent entries: long-term memory without a long context.

Turns are serialized: while one is being spoken, another arrival waits
(or is dropped if the queue is already full), so the pet never talks over
itself. What it hears reaches it through behavior/converse.py, which
decides whether it was being spoken to at all.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from ..events import SpeechEnded, SpeechStarted, StopRequested
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
        self._conversation_id: Optional[int] = None
        self._heard_this_episode = 0
        # Things that happened without a turn (reflexes), told with the next one.
        self._notes: list[str] = []
        self._hushed = False
        self.busy = False

    async def start(self) -> None:
        subs = [self.pet.bus.subscribe(StopRequested),
                self.pet.bus.subscribe(SpeechStarted, SpeechEnded)]
        self._tasks = [
            asyncio.create_task(self._watch_stops(subs[0]), name="brain-stops"),
            asyncio.create_task(self._feedback_loop(subs[1]), name="brain-feedback"),
            asyncio.create_task(self._run_turns(), name="brain-turns"),
        ]

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        # No journal entry on shutdown: it would hold Ctrl-C up for a model turn.
        await self._end_episode(summary=None)

    # --- inputs ---------------------------------------------------------------

    async def _watch_stops(self, sub) -> None:
        async for event in sub:
            # The brain's own stop() tool is part of its turn; let it answer.
            if event.source != "tool":
                self.hush()

    def hush(self) -> None:
        """Silences the rest of the current turn. The model still finishes it
        (its output has to be read either way); nothing more is said or done."""
        if self.busy:
            self._hushed = True

    def note(self, text: str) -> None:
        """Something the model should know at its next turn, without a turn of its own."""
        self._notes.append(text)

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
            try:
                text, kind, speaker = await asyncio.wait_for(self._queue.get(), self._idle_left())
            except asyncio.TimeoutError:
                # Summarizing here, in the only task that runs turns, means a
                # new turn can't race the episode's last one.
                await self._close_idle_episode()
                continue
            try:
                await self._run_turn(text, kind, speaker)
            except Exception:  # noqa: BLE001 - one bad turn must not kill the brain
                log.exception("turn failed")

    def _idle_left(self) -> Optional[float]:
        if not self._episode_open:
            return None
        return max(0.0, self._last_turn_at + self.cfg.episode_idle_timeout_s - time.time())

    async def _ensure_episode(self) -> None:
        if self._episode_open:
            return
        await self.backend.start_episode(prompt_module.build_system_prompt(self.pet))
        self._episode_open = True
        self._heard_this_episode = 0
        if self.pet.db is not None:
            self._conversation_id = self.pet.db.start_conversation()

    async def _close_idle_episode(self) -> None:
        log.info("episode idle for %.0fs; closing it", time.time() - self._last_turn_at)
        summary = None
        if self._heard_this_episode:
            try:
                summary = await asyncio.wait_for(self._summarize(), 60)
            except Exception:  # noqa: BLE001 - a lost journal line is not worth more
                log.exception("journal summary failed")
        await self._end_episode(summary)

    async def _summarize(self) -> Optional[str]:
        """Asks the model for its journal line. Collected, never spoken."""
        text = ""
        async for event in self.backend.send(prompt_module.JOURNAL_REQUEST):
            if isinstance(event, TextDelta):
                text += event.text
            elif isinstance(event, BrainError):
                log.warning("journal summary: %s", event.message)
        summary = prompt_module.clean_summary(text)
        log.info("journal: %s", summary)
        return summary or None

    async def _end_episode(self, summary: Optional[str]) -> None:
        if self._conversation_id is not None and self.pet.db is not None:
            self.pet.db.end_conversation(self._conversation_id, summary)
        self._conversation_id = None
        self._episode_open = False
        await self.backend.end_episode()

    async def _run_turn(self, text: str, kind: str, speaker: Optional[str]) -> None:
        await self._ensure_episode()
        notes, self._notes = self._notes, []
        turn = prompt_module.build_turn(self.pet, text, kind=kind, speaker=speaker, notes=notes)
        log.info("-> brain: %s", turn.replace("\n", " | "))

        if kind == "heard":
            self._heard_this_episode += 1
        self._record((speaker or "someone") if kind == "heard" else "event", text)

        parser = SpeechStreamParser()
        utterance = None
        said: list[str] = []
        self._hushed = False
        self.busy = True
        started = time.monotonic()
        try:
            async for event in self.backend.send(turn):
                if isinstance(event, TextDelta):
                    log.debug("delta %r", event.text)
                    for piece in parser.feed(event.text):
                        utterance = await self._emit(piece, utterance, said)
                elif isinstance(event, ToolStarted):
                    log.info("brain calls %s(%s)", event.name, event.arguments)
                elif isinstance(event, ToolFinished):
                    if event.is_error:
                        log.info("tool reported an error back to the brain")
                elif isinstance(event, BrainError):
                    log.error("brain error: %s", event.message)
                elif isinstance(event, TurnDone):
                    for piece in parser.flush():
                        utterance = await self._emit(piece, utterance, said)
                    log.info("turn done in %.1fs (cost %s)",
                             event.duration_s or (time.monotonic() - started), event.cost_usd)
        finally:
            self.busy = False
            self._last_turn_at = time.time()
            if said:
                self._record("pet", " ".join(said))
            if utterance is not None:
                utterance.end()
                await utterance.wait()
            if self.expressions is not None:
                await self.expressions.rest()

    def _record(self, speaker: str, text: str) -> None:
        if self._conversation_id is not None and self.pet.db is not None:
            self.pet.db.add_utterance(self._conversation_id, speaker, text)

    async def _emit(self, piece, utterance, said: list):
        if self._hushed:
            return utterance
        if isinstance(piece, Action):
            if self.expressions is not None:
                await self.expressions.apply(piece)
            return utterance
        if isinstance(piece, Sentence):
            log.info("<- brain says: %s", piece.text)
            said.append(piece.text)
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
