"""
Configuration: typed dataclasses with defaults, overlaid by a YAML file,
overlaid by environment variables.

Env overrides use `PETD__<SECTION>__<KEY>=<yaml value>`, e.g.
`PETD__FACE__HOST=192.168.101.41` or `PETD__STT__THREADS=4`.

Relative paths in the config are resolved against the project root
(the robot-pet directory), not the current working directory.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(Exception):
    pass


@dataclass
class PetConfig:
    name: str = "GLaDOS"


@dataclass
class NetworkConfig:
    # IP the face should stream audio to. "auto" picks the local address
    # used to reach the face.
    pc_ip: str = "auto"


@dataclass
class VacuumConfig:
    enabled: bool = True
    host: str = "192.168.101.43"
    port: int = 80
    poll_interval_s: float = 2.0
    # Resend interval for Valetudo's manual-control dead-man's switch.
    drive_update_interval_s: float = 0.15


@dataclass
class PlayerConfig:
    port: int = 6665


@dataclass
class FaceConfig:
    enabled: bool = True
    host: str = "192.168.101.40"
    port: int = 80
    audio_port: int = 5000
    audio_gain: Optional[float] = None       # None = leave the device setting alone
    enable_recognition: bool = True
    status_poll_s: float = 0.5
    # A face set going empty only counts as "nobody here" after this long,
    # to ride out detection dropouts (over 1.5 s with a face held still in
    # view; 3 s still lost a seated person several times a minute).
    faces_lost_debounce_s: float = 5.0


@dataclass
class SttConfig:
    enabled: bool = True
    binary: str = "stt/udp-stream/whisper-udp-stream"
    cwd: str = "stt/udp-stream"
    model: str = "models/ggml-base.en.bin"
    vad_model: str = "models/ggml-silero-v6.2.0.bin"
    threads: int = 8
    extra_args: list = field(default_factory=list)
    max_no_speech_prob: float = 0.6
    # Whisper's initial prompt. Without it "GLaDOS" comes out as "Gladys",
    # "G let us" or "Clovis". None = the pet's name.
    prompt: Optional[str] = None
    min_chars: int = 2
    # Compared after lowercasing and stripping punctuation.
    ignore_phrases: list = field(default_factory=lambda: [
        "you", "thank you", "thanks for watching", "bye", "okay",
    ])


@dataclass
class SpeakerConfig:
    enabled: bool = True
    sink: str = "robot"                       # robot | null
    robot_port: int = 6000                    # socat -> aplay on the vacuum
    # Connecting here kills aplay on the robot, for an instant interrupt():
    # socat -u TCP-LISTEN:6001,reuseaddr,fork EXEC:'killall aplay'
    # None = no such listener; interrupting then takes ~2 s to go quiet.
    stop_port: Optional[int] = None
    sample_rate: int = 22050                  # must match the robot's aplay -r
    # Output loudness. The robot's amixer controls nothing and piper's HTTP
    # server has no volume knob, so the PCM is scaled here. 1.0 = as piper
    # made it (peak-normalised, i.e. as loud as the speaker goes).
    volume: float = 0.2
    piper_url: str = "http://127.0.0.1:5001"
    manage_piper: bool = True                 # start piper.http_server if not already up
    piper_cwd: str = "piper-tts"
    piper_cmd: list = field(default_factory=lambda: [
        ".venv/bin/python", "-m", "piper.http_server",
        "--port", "5001", "-m", "glados_piper_medium.onnx",
    ])
    # How far ahead of real time audio is pushed to the robot: the jitter
    # buffer that rides out its WiFi stalls (measured up to 1.4 s). It costs
    # nothing on interrupt, which resets the connection and drops the lot.
    lead_s: float = 2.0
    # Rough delay between writing audio and hearing it (aplay startup + buffer).
    playback_latency_s: float = 0.3
    # Extra time after playback during which the mic hears our own echo.
    # scripts/echo_timing.py on the robot: the voice ends up to ~1.0 s after
    # its last byte was due, so latency + tail must be >= 1.3 s.
    gate_tail_s: float = 1.0


@dataclass
class BrainConfig:
    enabled: bool = True
    backend: str = "claude_cli"               # claude_cli | ollama
    model: str = "claude-haiku-4-5-20251001"  # latency-first; Sonnet 5 for more depth
    claude_binary: str = "claude"
    python: str = ".venv/bin/python"          # interpreter for the MCP shim
    runtime_dir: str = "runtime/brain"        # generated system.md + mcp.json, claude's cwd
    memory_dir: str = "memory"                # persona/backstory/body/style markdown
    extra_args: list = field(default_factory=list)
    # A conversation ends after this much quiet, so context stays small.
    episode_idle_timeout_s: float = 600.0
    turn_timeout_s: float = 120.0
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    max_tool_iterations: int = 8


@dataclass
class ConverseConfig:
    # Whisper spellings of the pet's name that count as being addressed, on
    # top of the name itself and anything one edit away from it.
    wake_words: list = field(default_factory=lambda: [
        "glados", "gladys", "gladis", "gladdis", "glad os", "glados's",
    ])
    # After the pet speaks, or after being addressed, it keeps listening
    # without its name for this long.
    window_s: float = 20.0
    # A known person in view counts as talking to the pet.
    gaze_opens: bool = True
    # Motor noise makes junk transcripts; while driving, only reflexes and
    # speech that names the pet get through.
    ignore_while_driving: bool = True


@dataclass
class MemoryConfig:
    enabled: bool = True
    db_path: str = "runtime/pet.db"
    # Greet a known person by name at most this often.
    greet_every_h: float = 4.0
    # Mention a stranger in view at most this often, and only once they've
    # been in view this long without being recognized. Recognizing someone
    # can take several seconds; at 4 s the pet asked Robin who they were.
    stranger_every_min: float = 30.0
    stranger_after_s: float = 10.0
    # familiarity tier N is reached at thresholds[N-1] = [interactions, days seen].
    familiarity_thresholds: list = field(default_factory=lambda: [[1, 1], [10, 3], [40, 10]])
    # The device's face slots (face_id_save_number in the firmware).
    face_slots: int = 7
    enroll_timeout_s: float = 6.0
    journal_in_prompt: int = 5
    facts_in_prompt: int = 15


@dataclass
class ApiConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class Config:
    pet: PetConfig = field(default_factory=PetConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    vacuum: VacuumConfig = field(default_factory=VacuumConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    converse: ConverseConfig = field(default_factory=ConverseConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def path(self, relative: str) -> Path:
        """Resolves a config path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _apply(obj: Any, data: dict, where: str) -> None:
    hints = get_type_hints(type(obj))
    for key, value in data.items():
        if not hasattr(obj, key) or key not in hints:
            raise ConfigError(f"unknown config key {where}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}{key} must be a mapping")
            _apply(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def _env_overrides(environ: dict) -> dict:
    out: dict = {}
    for name, raw in environ.items():
        if not name.startswith("PETD__"):
            continue
        parts = [p.lower() for p in name[len("PETD__"):].split("__")]
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: Optional[Path] = None, environ: Optional[dict] = None) -> Config:
    """
    Defaults <- YAML file (if given / found) <- PETD__* environment.
    With no explicit path, uses ./config.yaml in the project root if it exists.
    """
    cfg = Config()
    if path is None:
        candidate = PROJECT_ROOT / "config.yaml"
        path = candidate if candidate.exists() else None
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        _apply(cfg, data, "")
    _apply(cfg, _env_overrides(os.environ if environ is None else environ), "")
    return cfg
