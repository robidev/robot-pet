import importlib.util
import socket
import threading
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("mic_levels", Path(__file__).parents[1] / "scripts" / "mic_levels.py")
mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mic)

RATE = 16000


def tone(seconds, amplitude, rng):
    """A noise-like signal with this RMS (in counts)."""
    return rng.normal(0, amplitude, int(seconds * RATE))


def test_levels_tell_speech_from_the_floor():
    rng = np.random.default_rng(5)
    quiet = tone(2.0, 2.3, rng)                  # ~-83 dBFS: the mic's own hiss
    speech = tone(1.0, 33.0, rng)                # ~-60 dBFS: speech at 1 m
    samples = np.concatenate([quiet, speech, quiet]).round().astype(np.int16)
    got = mic.levels(samples, RATE)
    assert abs(got["floor_dbfs"] - mic.dbfs(2.3)) < 3
    assert abs(got["speech_dbfs"] - mic.dbfs(33.0)) < 1.5
    assert 0.8 <= got["speech_s"] <= 1.2
    assert got["clipped"] == 0 and got["seconds"] == 5.0


def test_clipping_is_counted():
    samples = np.array([0, 32767, -32768, 100] * 400, dtype=np.int16)
    assert mic.levels(samples, RATE)["clipped"] == 800


def test_the_face_packets_are_read_and_gaps_counted():
    port = 5099
    frames = [np.full(480, i, dtype="<i2") for i in range(6)]

    def send():
        time.sleep(0.2)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for seq, frame in enumerate(frames):
                if seq == 3:
                    continue                    # one lost on the way
                header = mic.HEADER.pack(mic.MAGIC, 1, 1, len(frame), seq, seq * 480, RATE)
                s.sendto(header + frame.tobytes(), ("127.0.0.1", port))
            s.sendto(b"junk", ("127.0.0.1", port))

    threading.Thread(target=send, daemon=True).start()
    samples, rate, lost = mic.record(port, 0.6)
    assert rate == RATE and lost == 1
    assert list(np.unique(samples)) == [0, 1, 2, 4, 5]
