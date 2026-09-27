"""
How loud is speech at the face's microphone, against the room's quiet?
(PLAN.md, Step 2: the microphone levels.) Records the face's UDP audio stream
(the same 16 kHz, 16-bit packets petd hears), saves each recording as a WAV and
prints its levels in dBFS:

    .venv/bin/python scripts/mic_levels.py quiet [--seconds 60]     # the room, nobody talking
    .venv/bin/python scripts/mic_levels.py say 2m [--seconds 8] [--text "come here glados"]
    .venv/bin/python scripts/mic_levels.py report                   # every recording so far

`say` records one sentence spoken at a distance (the label, e.g. 0.5m, 2m) and
transcribes it with whisper-cli, with petd's models and name prompt: was it
understood? Levels per recording: the noise floor (the quietest 10% of 100 ms
frames), speech (the frames 10 dB or more above the floor), the peak, clipping,
the signal-to-noise ratio against the latest `quiet`, and lost packets.

Recordings go to runtime/mic/<date>/<label>-<time>.wav, their numbers to
runtime/mic/levels.jsonl. It listens on petd's audio port (face.audio_port,
5000) and points the face there, so it refuses to run while petd is running.
Another --port needs letting through Windows' firewall first: with WSL's
mirrored networking, a UDP port nobody opened receives nothing (2026-09-27).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "face-api"))

from petd.config import Config, load_config  # noqa: E402
from petd.net import local_ip_towards  # noqa: E402

HEADER = struct.Struct("<4sBBHIII")     # magic, version, channels, samples, sequence, timestamp, rate
MAGIC = b"LGA1"
FRAME_S = 0.1                           # level frames
SPEECH_ABOVE_FLOOR_DB = 10.0


def dbfs(rms: float) -> float:
    return 20 * math.log10(max(rms, 1e-9) / 32768.0)


def record(port: int, seconds: float) -> tuple[np.ndarray, int, int]:
    """(samples, sample rate, packets lost) from the face's stream."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    chunks, rate, lost, last_seq = [], 16000, 0, None
    start = last_packet = time.monotonic()
    end = start + seconds
    try:
        while (now := time.monotonic()) < end:
            sock.settimeout(max(0.05, min(2.0, end - now)))
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                if time.monotonic() - last_packet >= 2.0:
                    raise SystemExit("no audio from the face for 2 s: is it up, and is its audio pointed here?")
                continue
            last_packet = time.monotonic()
            if len(data) < HEADER.size:
                continue
            magic, _version, channels, count, seq, _stamp, rate = HEADER.unpack_from(data)
            if magic != MAGIC or channels != 1:
                continue
            if last_seq is not None and seq > last_seq + 1:
                lost += seq - last_seq - 1
            last_seq = seq
            chunks.append(np.frombuffer(data[HEADER.size:HEADER.size + count * 2], dtype="<i2"))
    finally:
        sock.close()
    return (np.concatenate(chunks) if chunks else np.zeros(0, np.int16)), rate, lost


def levels(samples: np.ndarray, rate: int) -> dict:
    x = samples.astype(np.float64)
    step = int(rate * FRAME_S)
    frames = x[: len(x) // step * step].reshape(-1, step) if len(x) >= step else x.reshape(1, -1)
    frame_rms = np.sqrt((frames ** 2).mean(axis=1))
    floor = float(np.percentile(frame_rms, 10))
    loud = frame_rms[frame_rms >= floor * 10 ** (SPEECH_ABOVE_FLOOR_DB / 20)]
    return {
        "seconds": round(len(x) / rate, 1),
        "floor_dbfs": round(dbfs(floor), 1),
        "all_dbfs": round(dbfs(float(np.sqrt((x ** 2).mean()))), 1),
        "speech_dbfs": round(dbfs(float(np.sqrt((loud ** 2).mean()))), 1) if len(loud) else None,
        "speech_s": round(len(loud) * FRAME_S, 1),
        "peak_dbfs": round(dbfs(float(np.abs(x).max())), 1) if len(x) else None,
        "clipped": int((np.abs(samples.astype(np.int32)) >= 32767).sum()),
    }


def transcribe(cfg: Config, wav: Path) -> str:
    cli = ROOT / "stt" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    if not cli.exists():
        return "(no whisper-cli: stt/README.md)"
    cwd = cfg.path(cfg.stt.cwd)
    cmd = [str(cli), "-m", str(cwd / cfg.stt.model), "-f", str(wav), "-nt", "-np",
           "--prompt", cfg.pet.name, "-t", str(cfg.stt.threads)]
    vad = cwd / cfg.stt.vad_model
    if vad.exists():
        cmd += ["--vad", "--vad-model", str(vad)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(whisper-cli failed: {exc})"
    return " ".join(out.split()) or "(nothing)"


def petd_running(cfg: Config) -> bool:
    try:
        with socket.create_connection((cfg.api.host, cfg.api.port), timeout=0.5):
            return True
    except OSError:
        return False


def save_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())


def latest_quiet(log: Path) -> dict | None:
    if not log.exists():
        return None
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    quiet = [r for r in rows if r["kind"] == "quiet"]
    return quiet[-1] if quiet else None


def take(cfg: Config, kind: str, label: str, seconds: float, port: int, text: str | None) -> None:
    from face_client import FaceApiClient
    if petd_running(cfg):
        sys.exit("petd is running: its hearing would lose the audio meanwhile (scripts/start.sh stop)")
    face = FaceApiClient(cfg.face.host, cfg.face.port, timeout=5.0)
    pc_ip = cfg.network.pc_ip if cfg.network.pc_ip != "auto" else local_ip_towards(cfg.face.host, cfg.face.port)
    face.set_audio_destination(pc_ip, port)
    try:
        if kind == "say":
            print(f"recording {seconds:.0f} s: say the sentence now" + (f' ("{text}")' if text else ""))
        else:
            print(f"recording {seconds:.0f} s of the room: nobody talk")
        samples, rate, lost = record(port, seconds)
    finally:
        face.set_audio_destination(pc_ip, cfg.face.audio_port)       # back to petd's port
    stamp = time.strftime("%Y%m%d-%H%M%S")
    wav = ROOT / "runtime" / "mic" / stamp[:8] / f"{label}-{stamp[9:]}.wav"
    save_wav(wav, samples, rate)
    row = {"kind": kind, "label": label, "t": stamp, "wav": str(wav.relative_to(ROOT)),
           "lost_packets": lost, **levels(samples, rate)}
    log = ROOT / "runtime" / "mic" / "levels.jsonl"
    if kind == "say":
        quiet = latest_quiet(log)
        if quiet and row["speech_dbfs"] is not None:
            row["snr_db"] = round(row["speech_dbfs"] - quiet["floor_dbfs"], 1)
        row["expected"] = text
        row["heard"] = transcribe(cfg, wav)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(row) + "\n")
    show([row])
    print(f"-> {wav.relative_to(ROOT)}")


def show(rows: list[dict]) -> None:
    print(f"{'label':10s} {'kind':5s} {'floor':>6s} {'speech':>7s} {'peak':>6s} {'snr':>5s} "
          f"{'speech s':>8s} {'clip':>4s} {'lost':>4s}  heard")
    for r in rows:
        f = lambda v: "-" if v is None else f"{v:.1f}"
        print(f"{r['label'][:10]:10s} {r['kind']:5s} {f(r['floor_dbfs']):>6s} {f(r.get('speech_dbfs')):>7s} "
              f"{f(r.get('peak_dbfs')):>6s} {f(r.get('snr_db')):>5s} {r['speech_s']:8.1f} {r['clipped']:4d} "
              f"{r['lost_packets']:4d}  {r.get('heard', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config (default: ./config.yaml if present)")
    parser.add_argument("--port", type=int, help="where to receive the audio (default: face.audio_port)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("quiet", help="the room with nobody talking: the noise floor")
    p.add_argument("--seconds", type=float, default=60.0)
    p = sub.add_parser("say", help="one sentence spoken at a distance")
    p.add_argument("label", help="e.g. 0.5m, 1m, 2m, 3m")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--text", help="what will be said, to compare with what whisper heard")
    sub.add_parser("report", help="every recording so far")
    args = parser.parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    if args.cmd == "report":
        log = ROOT / "runtime" / "mic" / "levels.jsonl"
        if not log.exists():
            sys.exit("no recordings yet")
        show([json.loads(line) for line in log.read_text().splitlines() if line.strip()])
    else:
        take(cfg, args.cmd, "quiet" if args.cmd == "quiet" else args.label, args.seconds,
             args.port or cfg.face.audio_port,
             getattr(args, "text", None))


if __name__ == "__main__":
    main()
