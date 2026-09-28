from petd.brain.claude_cli import ClaudeCliBackend
from petd.config import Config


def test_thinking_off_is_the_clis_own_switch():
    cfg = Config()
    env = ClaudeCliBackend(cfg)._env()
    assert env["MAX_THINKING_TOKENS"] == "0" and "PETD_API" in env    # off by default
    cfg.brain.thinking = True
    assert "MAX_THINKING_TOKENS" not in ClaudeCliBackend(cfg)._env()


async def test_the_reply_the_cli_asks_for_after_a_silence_is_not_spoken():
    # Captured 2026-09-28 (CLI 2.1.283): the model said its line, called a
    # tool, then stayed silent; the CLI asked it for visible output, and it
    # said the line again.
    import asyncio
    import json
    from types import SimpleNamespace

    from petd.brain.backend import TextDelta, ToolStarted, TurnDone

    def delta(text):
        return {"type": "stream_event", "event": {"type": "content_block_delta",
                                                  "delta": {"type": "text_delta", "text": text}}}
    lines = [
        delta("I'm already there."),
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__robot__get_senses", "input": {}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "{}"}]}]}},
        {"type": "user", "message": {"content": [{"type": "text", "text":
            "[Your previous response had no visible output. Please continue and produce a "
            "user-visible response.]"}]}},
        delta("I'm already there."),
        {"type": "result", "subtype": "success", "result": "I'm already there."},
    ]
    stdout = asyncio.StreamReader()
    stdout.feed_data("".join(json.dumps(m) + "\n" for m in lines).encode())
    backend = ClaudeCliBackend(Config())
    backend._proc = SimpleNamespace(stdout=stdout)
    events = [e async for e in backend._read_turn(0.0)]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["I'm already there."]
    assert any(isinstance(e, ToolStarted) for e in events) and isinstance(events[-1], TurnDone)
