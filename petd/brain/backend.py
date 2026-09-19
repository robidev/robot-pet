"""
The LLM backend interface. Backends differ in how they run a turn
(a persistent Claude CLI process, an ollama HTTP loop, ...) but all of
them stream the same events back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolStarted:
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolFinished:
    name: str
    is_error: bool = False


@dataclass(frozen=True)
class TurnDone:
    text: str = ""
    cost_usd: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass(frozen=True)
class BrainError:
    message: str


BrainEvent = TextDelta | ToolStarted | ToolFinished | TurnDone | BrainError


@runtime_checkable
class LLMBackend(Protocol):
    async def start_episode(self, system_prompt: str) -> None:
        """Begins a fresh conversation (new context)."""

    def send(self, user_turn: str) -> AsyncIterator[BrainEvent]:
        """Runs one turn, streaming events as they happen."""

    async def end_episode(self) -> None:
        """Ends the conversation and releases any resources."""
