"""
Ollama backend: same interface as the Claude CLI one, for running the pet
on a local model later (PLAN.md D5).

It keeps the message history itself and runs its own tool loop against the
shared ToolRegistry (no MCP involved). Vision (the `look` tool's image)
needs a vision-capable model, e.g. qwen2.5-vl; with a text-only model the
image is dropped and only its text description is passed on.

Untested against a real ollama: nothing on this machine runs one yet.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from ..config import Config
from .backend import BrainError, BrainEvent, TextDelta, ToolFinished, ToolStarted, TurnDone
from .tools import ToolRegistry

log = logging.getLogger(__name__)


class OllamaBackend:
    def __init__(self, cfg: Config, registry: ToolRegistry):
        self.cfg = cfg.brain
        self.registry = registry
        self.url = cfg.brain.ollama_url
        self.messages: list[dict] = []
        self._client: Optional[httpx.AsyncClient] = None

    async def start_episode(self, system_prompt: str) -> None:
        await self.end_episode()
        self._client = httpx.AsyncClient(base_url=self.url, timeout=120.0)
        self.messages = [{"role": "system", "content": system_prompt}]

    async def end_episode(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _tool_specs(self) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["schema"]}}
                for t in self.registry.list()]

    async def send(self, user_turn: str) -> AsyncIterator[BrainEvent]:
        if self._client is None:
            yield BrainError("no episode started")
            return
        self.messages.append({"role": "user", "content": user_turn})
        spoken: list[str] = []

        for _ in range(self.cfg.max_tool_iterations):
            calls: list[dict] = []
            content = ""
            try:
                async with self._client.stream("POST", "/api/chat", json={
                    "model": self.cfg.ollama_model, "messages": self.messages,
                    "tools": self._tool_specs(), "stream": True,
                }) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        message = chunk.get("message", {})
                        if message.get("content"):
                            content += message["content"]
                            yield TextDelta(message["content"])
                        calls += message.get("tool_calls", [])
            except Exception as exc:  # noqa: BLE001
                yield BrainError(f"ollama: {exc}")
                return

            self.messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            spoken.append(content)
            if not calls:
                yield TurnDone(text="".join(spoken))
                return

            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                yield ToolStarted(name, arguments)
                result = await self.registry.call(name, arguments)
                yield ToolFinished(name, result.is_error)
                self.messages.append({"role": "tool", "name": name,
                                      "content": result.text or "done",
                                      **({"images": [result.image_b64]} if result.image_b64 else {})})

        yield BrainError(f"gave up after {self.cfg.max_tool_iterations} tool rounds")
