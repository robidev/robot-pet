"""Where the Claude CLI's time to first word goes: the brain's own argv, three turns."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from petd.brain.claude_cli import ClaudeCliBackend   # noqa: E402
from petd.config import load_config                 # noqa: E402

cfg = load_config(ROOT / "local_config.yaml")
backend = ClaudeCliBackend(cfg)
# argv: extra CLI args, and NAME=value for the environment
extra = [a for a in sys.argv[1:] if "=" not in a or a.startswith("--")]
env_extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))
argv = backend._argv() + extra
print("extra:", extra, env_extra)
t_spawn = time.monotonic()
proc = subprocess.Popen(argv, cwd=backend.runtime_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1,
                        env={**os.environ, "PETD_API": backend.api_url, **env_extra})
TURNS = ["We're testing how quickly you respond. Can you say something now?",
         "What is two plus two?",
         "Tell me one short thing about cake."]
time.sleep(5)          # petd prestarts the process long before anyone speaks
for text in TURNS:
    turn = f"[{time.strftime('%H:%M')} | battery 100% | docked | head offline]\nsomeone says: \"{text}\""
    msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": turn}]}}
    t0 = time.monotonic()
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()
    marks, blocks, reply = {}, [], ""
    for line in proc.stdout:
        t = time.monotonic() - t0
        m = json.loads(line)
        marks.setdefault("first output", t)
        if m["type"] == "stream_event":
            e = m["event"]
            if e["type"] == "message_start":
                marks.setdefault("message_start (API answered)", t)
                usage = e["message"].get("usage", {})
            elif e["type"] == "content_block_start":
                blocks.append((round(t, 2), e["content_block"]["type"]))
            elif e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta":
                marks.setdefault("first text", t)
                reply += e["delta"]["text"]
        elif m["type"] == "result":
            marks["result"] = t
            api_ms = m.get("duration_api_ms"); total_ms = m.get("duration_ms")
            break
    print(f"\n{text!r} -> {reply.strip()!r}")
    for k, v in marks.items():
        print(f"  {v:5.2f} s  {k}")
    print(f"  blocks: {blocks}")
    print(f"  CLI says: duration_ms {total_ms}, duration_api_ms {api_ms}; "
          f"input tokens {usage.get('input_tokens')}, cache read {usage.get('cache_read_input_tokens')}, "
          f"cache write {usage.get('cache_creation_input_tokens')}")
proc.stdin.close(); proc.wait(timeout=10)
