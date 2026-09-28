from petd.brain.claude_cli import ClaudeCliBackend
from petd.config import Config


def test_thinking_off_is_the_clis_own_switch():
    cfg = Config()
    assert "MAX_THINKING_TOKENS" not in ClaudeCliBackend(cfg)._env()
    cfg.brain.thinking = False
    env = ClaudeCliBackend(cfg)._env()
    assert env["MAX_THINKING_TOKENS"] == "0" and "PETD_API" in env
