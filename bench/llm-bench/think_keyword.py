"""Does a keyword in one message turn thinking on in a CLI started with thinking off?"""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from petd.brain.claude_cli import ClaudeCliBackend   # noqa: E402
from petd.config import load_config                 # noqa: E402

QUESTION = "Plan how you would find out which room the cat is in, then answer in one sentence."
CASES = [  # (label, environment, what is appended to the message)
    ("thinking off, no keyword", {"MAX_THINKING_TOKENS": "0"}, ""),
    ("thinking off + 'think hard'", {"MAX_THINKING_TOKENS": "0"}, " Think hard about this."),
    ("thinking off + 'ultrathink'", {"MAX_THINKING_TOKENS": "0"}, " ultrathink"),
    ("thinking on (default), no keyword", {}, ""),
]
backend = ClaudeCliBackend(load_config(ROOT / "local_config.yaml"))
for label, env, suffix in CASES:
    proc = subprocess.Popen(backend._argv(), cwd=backend.runtime_dir, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                            env={**os.environ, "PETD_API": backend.api_url, **env})
    time.sleep(4)
    turn = f"[15:00 | battery 100% | docked | head offline]\nsomeone says: \"{QUESTION}{suffix}\""
    t0 = time.monotonic()
    proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": turn}]}}) + "\n"); proc.stdin.flush()
    blocks, first, cost = [], None, None
    for line in proc.stdout:
        m = json.loads(line)
        if m["type"] == "stream_event":
            e = m["event"]
            if e["type"] == "content_block_start":
                blocks.append(e["content_block"]["type"])
            elif e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta" and first is None:
                first = time.monotonic() - t0
        elif m["type"] == "result":
            cost = m.get("total_cost_usd"); break
    proc.stdin.close(); proc.wait(timeout=10)
    print(f"{label:36s} blocks {blocks}, first word {first:.2f} s, ${cost:.4f}")
