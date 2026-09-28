"""
Is everything petd needs in place? (PLAN.md G2) Run by scripts/start.sh before
it starts petd, and on its own:

    .venv/bin/python scripts/preflight.py

Missing pieces petd can't run without (the speech recognizer and its models,
the voice, the Claude CLI, a free API port) are errors: exit code 1. Devices
that don't answer are warnings: petd waits for them and picks them up when they
come. Everything is read from the config, so disabled parts aren't checked.
"""

from __future__ import annotations

import argparse
import ipaddress
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from petd.config import Config, load_config  # noqa: E402

OK, WARN, ERROR = "ok", "warn", "error"


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def local_addresses() -> list[str]:
    try:
        out = subprocess.run(["ip", "-4", "-brief", "addr"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [word.split("/")[0] for line in out.splitlines() for word in line.split()[2:] if "/" in word]


def alsa_error(tool: str, device: str) -> str:
    """Why `tool` (aplay | arecord) can't open an ALSA device, or "" if it can."""
    rate = "16000" if tool == "arecord" else "22050"
    argv = [tool, "-q", "-D", device, "-t", "raw", "-f", "S16_LE", "-r", rate, "-c", "1"]
    if tool == "arecord":
        argv += ["-s", "160"]               # 10 ms and done
    try:
        result = subprocess.run([*argv, "/dev/null"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if result.returncode == 0:
        return ""
    lines = result.stderr.strip().splitlines()
    return lines[-1] if lines else f"exit code {result.returncode}"


def checks(cfg: Config) -> list[tuple[str, str, str]]:
    """[(status, what, detail)]"""
    found: list[tuple[str, str, str]] = []

    def check(what: str, ok: bool, detail: str, failed: str = ERROR) -> None:
        found.append((OK if ok else failed, what, "" if ok else detail))

    path = cfg.path
    check("config", (ROOT / "config.yaml").exists(),
          "no config.yaml: running on the defaults (cp config.example.yaml config.yaml)", WARN)

    if cfg.api.enabled:
        check("petd not already running", not tcp_open(cfg.api.host, cfg.api.port, 0.5),
              f"something listens on {cfg.api.host}:{cfg.api.port}: petd is running already "
              "(scripts/start.sh stop)")

    if cfg.stt.enabled:
        binary = path(cfg.stt.binary)
        check("speech recognizer", binary.exists(), f"{binary} missing: build it (stt/README.md)")
        for model in (cfg.stt.model, cfg.stt.vad_model):
            model_path = path(cfg.stt.cwd) / model
            check(f"model {Path(model).name}", model_path.exists(),
                  f"{model_path} missing: download it (stt/README.md)")
        if cfg.stt.source == "local":
            err = alsa_error("arecord", cfg.stt.local_device)
            check(f"microphone {cfg.stt.local_device}", not err,
                  f"{err}: the pet will be deaf (plugged in, attached to WSL, in the audio group?)", WARN)

    if cfg.speaker.enabled and cfg.speaker.sink == "local":
        err = alsa_error("aplay", cfg.speaker.local_device)
        check(f"speaker {cfg.speaker.local_device}", not err,
              f"{err}: the pet will be silent (plugged in, attached to WSL, in the audio group?)", WARN)

    if cfg.speaker.enabled and cfg.speaker.manage_piper:
        cwd = path(cfg.speaker.piper_cwd)
        check("voice (piper)", (cwd / cfg.speaker.piper_cmd[0]).exists(),
              f"{cwd / cfg.speaker.piper_cmd[0]} missing: set up piper (piper-tts/README.md)")
        for arg in cfg.speaker.piper_cmd:
            if arg.endswith(".onnx"):
                check("voice model", (cwd / arg).exists(), f"{cwd / arg} missing (piper-tts/README.md)")

    if cfg.brain.enabled and cfg.brain.backend == "claude_cli":
        check("Claude CLI", shutil.which(cfg.brain.claude_binary) is not None,
              f"`{cfg.brain.claude_binary}` not on PATH: install Claude Code and log in once")

    if cfg.recognition.enabled:
        models = path(cfg.recognition.models_dir)
        have = all((models / m).exists() for m in (cfg.recognition.detector_model,
                                                   cfg.recognition.recognizer_model))
        check("face recognition models", have,
              "missing: nobody will be recognized (scripts/fetch_face_models.py)", WARN)

    # Networking: the face's audio only reaches WSL with mirrored networking.
    if shutil.which("wslinfo") and cfg.stt.enabled and cfg.stt.source == "face":
        try:
            mode = subprocess.run(["wslinfo", "--networking-mode"], capture_output=True, text=True,
                                  timeout=3).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            mode = "unknown"
        check("WSL mirrored networking", mode == "mirrored",
              f"networking mode is {mode or 'unknown'}: the face's audio won't reach petd "
              "(README.md, setup step 1)", WARN)

    hosts = [h for h, on in ((cfg.face.host, cfg.face.enabled), (cfg.vacuum.host, cfg.vacuum.enabled)) if on]
    if hosts:
        mine = local_addresses()
        nets = {ipaddress.ip_network(f"{h}/24", strict=False) for h in hosts}
        check("PC on the devices' network", any(ipaddress.ip_address(a) in n for a in mine for n in nets),
              f"none of this PC's addresses ({', '.join(mine) or 'none'}) is on "
              f"{', '.join(map(str, nets))}: another network?", WARN)

    if cfg.face.enabled:
        check(f"face {cfg.face.host}", tcp_open(cfg.face.host, cfg.face.port),
              "not answering: off, rebooting, or its WiFi is stuck (petd waits for it)", WARN)
    if cfg.vacuum.enabled:
        check(f"robot {cfg.vacuum.host} (Valetudo)", tcp_open(cfg.vacuum.host, cfg.vacuum.port),
              "not answering (petd waits for it)", WARN)
        if cfg.speaker.enabled and cfg.speaker.sink == "robot":
            check("robot speaker", tcp_open(cfg.vacuum.host, cfg.speaker.robot_port),
                  f"nothing on port {cfg.speaker.robot_port}: the pet will be silent (speaker.sh)", WARN)
        check("robot Player", tcp_open(cfg.vacuum.host, cfg.player.port),
              f"nothing on port {cfg.player.port}: no odometry for driving", WARN)
    return found


def report(found: list[tuple[str, str, str]], out: Callable[[str], None] = print) -> int:
    marks = {OK: "  ok  ", WARN: " warn ", ERROR: "ERROR "}
    for status, what, detail in found:
        out(f"{marks[status]} {what}" + (f": {detail}" if detail else ""))
    errors = sum(status == ERROR for status, _, _ in found)
    warnings = sum(status == WARN for status, _, _ in found)
    if errors:
        out(f"{errors} error(s): petd would not work; fix these first")
    elif warnings:
        out(f"ready, with {warnings} warning(s)")
    else:
        out("ready")
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config (default: ./config.yaml if present)")
    args = parser.parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    sys.exit(report(checks(cfg)))


if __name__ == "__main__":
    main()
