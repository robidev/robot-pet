"""
Claude CLI backend: one persistent `claude -p` process per episode,
talking stream-json over stdin/stdout.

Isolation matters here. The pet runs with its own cwd, no inherited
settings or CLAUDE.md, no built-in tools (no Bash, no file access) and
only the robot's MCP tools allowed, so the persona can't be talked into
touching the machine petd runs on.

Protocol (verified against CLI 2.1.278):
  in:  {"type":"user","message":{"role":"user","content":[{"type":"text","text":...}]}}
  out: {"type":"system","subtype":"init",...}            once per turn
       {"type":"stream_event","event":{...}}             deltas (thinking + text)
       {"type":"assistant","message":{...}}              complete blocks, incl. tool_use
       {"type":"result","subtype":"success","result":..., "total_cost_usd":...}

When the model stays silent after a tool (as memory/style.md asks when
there's nothing new to say), the CLI tells it "[Your previous response had
no visible output. ...]", and it repeats its last line or explains that it
already spoke (2026-09-28, CLI 2.1.283). That reply is not spoken.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from ..config import BrainConfig, Config

# The CLI's own message to the model after a reply with no text.
_NUDGE = "Your previous response had no visible output"
from .backend import BrainError, BrainEvent, TextDelta, ToolFinished, ToolStarted, TurnDone

log = logging.getLogger(__name__)


class ClaudeCliBackend:
    def __init__(self, cfg: Config):
        self.cfg: BrainConfig = cfg.brain
        self.runtime_dir: Path = cfg.path(self.cfg.runtime_dir)
        self.api_url = f"http://{cfg.api.host}:{cfg.api.port}"
        self.project_root = cfg.path(".")
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._session_id: Optional[str] = None
        self._lock = asyncio.Lock()

    # --- episode lifecycle ----------------------------------------------------

    async def start_episode(self, system_prompt: str) -> None:
        await self.end_episode()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "system.md").write_text(system_prompt)
        self._write_mcp_config()
        argv = self._argv()
        log.info("starting brain: %s", " ".join(argv[:6]) + " ...")
        self._proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.runtime_dir,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
            # A tool result carrying a photo comes back as one base64 line,
            # far past asyncio's 64 KiB readline default.
            limit=64 * 1024 * 1024,
            env=self._env(),
        )
        asyncio.create_task(self._log_stderr(), name="brain-stderr")

    async def end_episode(self) -> None:
        proc, self._proc = self._proc, None
        self._session_id = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), 5)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
            proc.kill()
            await proc.wait()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _write_mcp_config(self) -> None:
        """MCP config naming the stdio shim that forwards to petd's HTTP API."""
        # Absolute: the MCP client may resolve a relative command itself.
        python = Path(self.cfg.python)
        if not python.is_absolute():
            python = self.project_root / python
        config = {"mcpServers": {"robot": {
            "command": str(python),
            "args": ["-m", "petd.mcp_shim.robot_mcp", "--api", self.api_url],
            "cwd": str(self.project_root),
            "env": {"PYTHONPATH": str(self.project_root)},
        }}}
        (self.runtime_dir / "mcp.json").write_text(json.dumps(config, indent=2))

    def _env(self) -> dict:
        env = {**os.environ, "PETD_API": self.api_url}
        if not self.cfg.thinking:
            env["MAX_THINKING_TOKENS"] = "0"      # the CLI's switch for extended thinking
        return env

    def _argv(self) -> list[str]:
        c = self.cfg
        argv = [
            c.claude_binary, "-p",
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--verbose", "--include-partial-messages",
            "--model", c.model,
            "--system-prompt-file", str(self.runtime_dir / "system.md"),
            "--mcp-config", str(self.runtime_dir / "mcp.json"), "--strict-mcp-config",
            # No built-in tools: the pet gets the robot's abilities, not the PC's.
            "--tools", "",
            "--allowedTools", "mcp__robot__*",
            # Don't inherit the user's settings, CLAUDE.md or memories.
            "--setting-sources", "",
        ]
        return argv + list(c.extra_args)

    async def _log_stderr(self) -> None:
        proc = self._proc
        if proc is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                log.warning("claude: %s", text)

    # --- one turn -------------------------------------------------------------

    async def send(self, user_turn: str) -> AsyncIterator[BrainEvent]:
        async with self._lock:
            if not self.running:
                yield BrainError("brain process is not running")
                return
            started = time.monotonic()
            message = {"type": "user", "message": {"role": "user",
                                                   "content": [{"type": "text", "text": user_turn}]}}
            try:
                self._proc.stdin.write((json.dumps(message) + "\n").encode())
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                yield BrainError(f"brain process died: {exc}")
                return

            async for event in self._read_turn(started):
                yield event

    async def _read_turn(self, started: float) -> AsyncIterator[BrainEvent]:
        assert self._proc is not None
        nudged = False              # the CLI asked for words after the model chose silence
        while True:
            try:
                line = await asyncio.wait_for(self._proc.stdout.readline(), self.cfg.turn_timeout_s)
            except asyncio.TimeoutError:
                yield BrainError(f"no reply within {self.cfg.turn_timeout_s}s")
                return
            if not line:
                yield BrainError("brain process closed its output")
                return
            try:
                msg = json.loads(line)
            except ValueError:
                continue

            kind = msg.get("type")
            if kind == "stream_event":
                event = msg.get("event", {})
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    # thinking_delta / signature_delta are internal: not speech.
                    if delta.get("type") == "text_delta" and delta.get("text") and not nudged:
                        yield TextDelta(delta["text"])
            elif kind == "assistant":
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use":
                        yield ToolStarted(_short_tool_name(block.get("name", "?")),
                                          block.get("input", {}))
            elif kind == "user":
                # Tool results come back as a synthetic user message.
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "tool_result":
                        yield ToolFinished("", bool(block.get("is_error")))
                    elif block.get("type") == "text" and _NUDGE in block.get("text", ""):
                        log.info("the CLI asked for words after a silent reply: not speaking them")
                        nudged = True
            elif kind == "system" and msg.get("subtype") == "init":
                self._session_id = msg.get("session_id")
            elif kind == "result":
                if msg.get("is_error"):
                    yield BrainError(str(msg.get("result", "unknown error")))
                yield TurnDone(text=msg.get("result") or "",
                               cost_usd=msg.get("total_cost_usd"),
                               duration_s=time.monotonic() - started)
                return


def _short_tool_name(name: str) -> str:
    """mcp__robot__look -> look"""
    return name.rsplit("__", 1)[-1]
