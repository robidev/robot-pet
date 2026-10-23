"""
whisper-udp-stream: transcription time and text, full 30 s window vs sized
audio_ctx, 8 vs 4 threads. Test speech is piper saying LINES (16 kHz),
sent as the mic's packets at real-time pace, with petd's VAD args and prompt.

    .venv/bin/python bench/whisper-bench/bench.py [threads:pad_ms ...] [--rounds N] [--model models/ggml-small.en.bin] [--engine parakeet]
"""
import json, math, socket, subprocess, sys, time, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from petd.io.stt import lga1_packet, MIC_PACKET_SAMPLES, MIC_RATE   # noqa: E402

HERE = Path(__file__).parent
PORT = 5099
LINES = [
    "Hello GLaDOS.",                      # warm-up, not counted
    "Hello GLaDOS.",
    "Can you go back to your dock please?",
    "We're testing how quickly you respond. Can you say something now?",
    "GLaDOS, what do you see in the kitchen right now?",
    "Stop.",
    "Tell me something interesting about the history of cake, and keep it short.",
    "Who is in the room with me?",
    "I was thinking that tomorrow morning we could clean the living room, but only after "
    "everyone has left for work, because the noise is quite annoying.",
]
VAD = ["--vad-threshold", "0.3", "--min-speech-ms", "150", "--min-silence-ms", "500",
       "--speech-pad-ms", "300", "--max-speech-s", "30"]


def speech(i: int, text: str) -> np.ndarray:
    wav = HERE / f"line{i}.wav"
    if not wav.exists():
        subprocess.run([str(ROOT / "piper-tts/.venv/bin/python"), "-m", "piper",
                        "-m", str(ROOT / "piper-tts/glados_piper_medium.onnx"), "-f", str(wav)],
                       input=text.encode(), check=True, capture_output=True)
    with wave.open(str(wav)) as w:
        rate, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2")
    t = np.arange(0, len(pcm) / rate, 1 / MIC_RATE)
    return np.interp(t, np.arange(len(pcm)) / rate, pcm.astype(float))


def run(threads: int, pad_ms: int, clips: list) -> list:
    """[(text, transcribe_ms) or None per clip], matched by when each clip was sent."""
    import queue, threading
    argv = [str(ROOT / "stt/udp-stream/whisper-udp-stream"), "--json", "--port", str(PORT),
            "--threads", str(threads), "--engine", ENGINE, "--model", MODEL,
            "--vad-model", "models/ggml-silero-v6.2.0.bin",
            *(["--prompt", "GLaDOS"] if ENGINE == "whisper" else []),
            "--audio-ctx-pad-ms", str(pad_ms), *VAD]
    proc = subprocess.Popen(argv, cwd=ROOT / "stt/udp-stream", stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(json.loads(l)) for l in proc.stdout], daemon=True).start()
    assert lines.get(timeout=30)["type"] == "ready"
    sock, seq = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 0
    rng = np.random.default_rng(0)
    gap = np.zeros(int(1.5 * MIC_RATE))
    spans, pos = [], len(gap)
    for c in clips:
        spans.append((pos / MIC_RATE, (pos + len(c)) / MIC_RATE))
        pos += len(c) + len(gap)
    audio = np.concatenate([gap] + [np.concatenate([c, gap]) for c in clips])
    audio = audio + rng.normal(0, 30, len(audio))                # a quiet room, ~-60 dBFS
    pcm = np.clip(audio, -32768, 32767).astype("<i2").tobytes()
    step = MIC_PACKET_SAMPLES * 2
    t0, wall0 = time.monotonic(), time.time()
    for n, i in enumerate(range(0, len(pcm) - step + 1, step)):
        time.sleep(max(0.0, t0 + n * MIC_PACKET_SAMPLES / MIC_RATE - time.monotonic()))
        sock.sendto(lga1_packet(seq, pcm[i:i + step]), ("127.0.0.1", PORT)); seq += 1
    out: list = [None] * len(clips)
    while True:
        try:
            msg = lines.get(timeout=30)              # big models lag well behind the audio
        except queue.Empty:
            break
        if msg["type"] != "text":
            continue
        t = msg["t_start_utc"] - wall0
        for k, (a, b) in enumerate(spans):
            if a - 1.0 <= t <= b:
                out[k] = (msg["text"].strip(), msg["transcribe_ms"])
    proc.terminate(); proc.wait()
    return out


clips = [speech(i, t) for i, t in enumerate(LINES)]
print("lines:", ", ".join(f"{len(c) / MIC_RATE:.1f}s" for c in clips[1:]))
# argv: threads:pad_ms ... [--rounds N]
args = sys.argv[1:]
ROUNDS = int(args[args.index("--rounds") + 1]) if "--rounds" in args else 2
MODEL = args[args.index("--model") + 1] if "--model" in args else "models/ggml-base.en.bin"
ENGINE = args[args.index("--engine") + 1] if "--engine" in args else "whisper"
print("engine:", ENGINE, "model:", MODEL)
CONFIGS = [tuple(int(x) for x in a.split(":")) for a in args if ":" in a] or [(8, -1), (8, 1000), (4, -1), (4, 1000)]
results = {c: [] for c in CONFIGS}
for round_ in range(ROUNDS):
    for c in CONFIGS:
        results[c].append(run(*c, clips))
        print(f"round {round_ + 1}: threads {c[0]}, pad {c[1]} done", flush=True)

for (threads, pad), rounds in results.items():
    label = f"threads {threads}, {'full 30 s window' if pad < 0 else f'utterance + {pad} ms'}"
    per_line = []
    for k in range(1, len(LINES)):
        got = [r[k] for r in rounds]
        per_line.append(got)
    ms = [g[1] for got in per_line for g in got if g]
    missing = sum(g is None for got in per_line for g in got)
    print(f"\n{label}: median {np.median(ms):.0f} ms, mean {np.mean(ms):.0f} ms, missing {missing}")
    for k, got in enumerate(per_line, 1):
        times = "/".join(f"{g[1]:4d}" if g else "   -" for g in got)
        texts = {g[0] for g in got if g}
        print(f"  {len(clips[k]) / MIC_RATE:4.1f}s  {times} ms  {' | '.join(texts)}")
(HERE / f"results-{Path(MODEL).stem}.json").write_text(json.dumps({f"{k[0]}/{k[1]}": v for k, v in results.items()}, indent=1))
