"""The bench's 8 lines as 16 kHz files: piper's wav resampled as bench.py does,
0.3 s either side (--speech-pad-ms), the bench's quiet-room noise (std 30)."""
import sys, wave
from pathlib import Path
import numpy as np
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(0)
for i in range(9):
    with wave.open(str(src / f"line{i}.wav")) as w:
        rate, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2")
    t = np.arange(0, len(pcm) / rate, 1 / 16000)
    x = np.interp(t, np.arange(len(pcm)) / rate, pcm.astype(float))
    pad = np.zeros(int(0.3 * 16000))
    x = np.concatenate([pad, x, pad]) + rng.normal(0, 30, len(x) + 2 * len(pad))
    with wave.open(str(out / f"clip{i}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())
    print(f"clip{i}.wav {len(x) / 16000:.1f}s (source {rate} Hz)")
