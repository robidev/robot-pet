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
    # Only after this long without an answer is the base offline: one missed
    # poll in a WiFi stall made the brain refuse a command (2026-09-23).
    offline_after_s: float = 10.0
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
    status_poll_s: float = 0.5
    offline_after_s: float = 10.0             # see VacuumConfig
    # A face set going empty only counts as "nobody here" after this long,
    # to ride out detection dropouts (over 1.5 s with a face held still in
    # view; 3 s still lost a seated person several times a minute).
    faces_lost_debounce_s: float = 5.0
    # The tilt mount's range: 67 (down) to 180 (up) since the servos were
    # remounted reversed (2026-10-05, the firmware's limits); before, 58 (up)
    # to 105 (down), measured by hand. Held against a stop, the servo browns
    # the board out. Firmware clamps too.
    tilt_min_deg: float = 67.0
    tilt_max_deg: float = 180.0
    # How far "left"/"right" and "up"/"down" turn the head from where it's
    # looking, for a [look:...] glance and look_direction alike. Asked to
    # look left, the brain sent pan 0, the servo's end (2026-10-05).
    look_turn_deg: float = 30.0
    look_tilt_deg: float = 15.0
    # A look asked for holds this long, then the head goes back to where it
    # was and tracking comes back on if it was on (it stared on, 2026-10-05).
    look_hold_s: float = 20.0


@dataclass
class SttConfig:
    enabled: bool = True
    binary: str = "stt/udp-stream/whisper-udp-stream"
    cwd: str = "stt/udp-stream"
    # Where the microphone is. face: the head's UDP stream (face.audio_port).
    # local: a sound card on this PC (an ALSA device, e.g. a USB speakerphone),
    # recorded with arecord and fed to whisper as the face's packets would be.
    source: str = "face"                      # face | local
    local_device: str = "default"             # arecord -D, for source: local
    # whisper's UDP port for source: local. Not face.audio_port: the face
    # keeps streaming there and the two would interleave.
    local_port: int = 5002
    model: str = "models/ggml-base.en.bin"
    vad_model: str = "models/ggml-silero-v6.2.0.bin"
    threads: int = 4                          # faster than 8 on this PC's 4 cores (PLAN.md)
    extra_args: list = field(default_factory=list)
    max_no_speech_prob: float = 0.6
    # An utterance is the pet's own echo when at least this share of it
    # overlaps its audible speech (plus gate_tail_s). Any overlap at all used
    # to count, which dropped answers begun as the pet's voice died away.
    echo_overlap: float = 0.5
    # The mic doesn't hear the pet's own voice (a speakerphone's echo
    # cancellation, e.g. the Jabra SPEAK 510). Then nothing is filtered as
    # echo, by time or by words (echo_overlap is unused), the pet's own
    # speech doesn't hide someone starting to talk, and a stop or quiet word
    # said over its voice interrupts it (converse.py).
    echo_cancelled: bool = False
    # Whisper's initial prompt. Without it "GLaDOS" comes out as "Gladys",
    # "G let us" or "Clovis". None = the pet's name.
    prompt: Optional[str] = None
    min_chars: int = 2
    # Compared after lowercasing and stripping punctuation.
    ignore_phrases: list = field(default_factory=lambda: [
        "you", "thank you", "thanks for watching", "thank you for watching", "bye", "okay",
    ])


@dataclass
class SpeakerConfig:
    enabled: bool = True
    # robot: socat -> aplay on the vacuum. local: aplay on this PC (e.g. a
    # USB speakerphone). null: nowhere.
    sink: str = "robot"                       # robot | local | null
    robot_port: int = 6000                    # socat -> aplay on the vacuum
    # aplay -D for sink: local. A plughw: device converts piper's 22050 Hz
    # mono to whatever the card plays.
    local_device: str = "default"
    # Connecting here kills aplay on the robot, for an instant interrupt()
    # (/root/watchdog_scripts/speaker_stop.sh, started by WatchDoge):
    # socat -u TCP-LISTEN:6001,reuseaddr,fork EXEC:'killall aplay'
    # None = no such listener; interrupting then takes ~2 s to go quiet.
    stop_port: Optional[int] = 6001
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
    # (A local sink has no WiFi to ride out; interrupting kills its aplay.)
    lead_s: float = 2.0
    # Rough delay between writing audio and hearing it (aplay startup + buffer).
    # This and gate_tail_s are the robot's; measure a local sink's with
    # scripts/echo_timing.py.
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
    # The model reasons privately before replying (claude_cli). Off, the
    # first word comes as soon as the API answers: 0.54-0.67 s against
    # 2.0-3.7 s with it, of which 1.4-2.8 s was thinking (2026-09-28, three
    # turns with the brain's own prompt). Six planning requests in --fake
    # mode got the same plans either way, 2-12 s sooner and at half the cost
    # without it (runtime/llm-bench/planning.py).
    thinking: bool = False
    # A conversation ends after this much quiet, so context stays small.
    episode_idle_timeout_s: float = 600.0
    turn_timeout_s: float = 120.0
    # Start the model's process before anyone speaks (at start-up, and after
    # a conversation closes), so the first reply isn't also waiting for it.
    prestart: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    max_tool_iterations: int = 8


@dataclass
class CalibrationConfig:
    """
    Measured conventions (PLAN.md 4.1, C3). Map heading: Valetudo's robot
    angle a (degrees) points along (sin a, -cos a) in map cm, i.e.
    theta = a + map_heading_offset_deg from +x towards +y, and it grows with
    a counter-clockwise turn, as odometry does (2026-09-22: a +90 turn moved
    it 45 -> 143, and 20 cm forward then moved the map pose +15/+15 cm).
    """
    map_heading_offset_deg: float = -90.0
    map_heading_sign: float = 1.0
    # The camera (spatial/person.py, C5). Names as scripts/calibrate_face.py fit
    # prints them. Until Step 2 measures them, camera_calibrated stays false and
    # these are starting values: K_face from E6a's snapshots (faces 195/80/47 px
    # of 480 at 0.6/1.5/2.5 m: distance x box height ~0.245 at all three, with
    # the PC's detector; the head's boxes may differ), the rest nominal.
    camera_calibrated: bool = False
    camera_height_m: float = 0.20             # the lens above the floor; the head's design may change it
    camera_forward_cm: float = 0.0            # the lens ahead of the robot's map position
    K_face: float = 0.245                     # horizontal distance (m) x face box height (fraction of the frame)
    hfov_deg: float = 60.0
    vfov_deg: Optional[float] = None          # None: from hfov_deg for a 4:3 frame
    pan_forward_deg: float = 75.0             # pan 75 = straight ahead, the firmware's centre (2026-10-05)
    # The signs as measured on 2026-10-05, after the servos were remounted
    # reversed: pan grows to the robot's right, a still scene moves left in
    # the image as pan grows, and higher tilt looks up (all three were +1).
    pan_sign: float = -1.0                    # +1: pan grows to the robot's left
    cx_per_pan_deg_sign: float = -1.0         # +1: a face moves right in the image as pan grows
    tilt_level_deg: float = 90.0              # tilt 90 is level, before and after the remount (2026-10-05)
    tilt_deg_per_elevation_deg: float = -1.0  # +1: lower tilt looks up


@dataclass
class PersonConfig:
    """Where a person is, from what the head sees (spatial/person.py, C5; PLAN.md 4.2)."""
    standoff_m: float = 1.0                   # stop this far in front of them
    clearance_cm: float = 25.0                # the approach target this far from anything mapped
    # Eye height by posture, when the person's own (people.face_z_m) isn't known:
    # the one whose tilt distance agrees best with the face-size distance wins.
    postures: dict = field(default_factory=lambda: {"standing": 1.55, "sitting or a child": 1.2})
    min_elevation_deg: float = 5.0            # below: tilt says little about distance, size only
    tilt_weight: float = 0.5                  # tilt vs size distance, when both are usable


@dataclass
class MotionConfig:
    """Closed-loop turn/move over Valetudo manual control (spatial/motion.py)."""
    enabled: bool = True
    # Moves are ignored until the lidar has spun up after arming (~6 s).
    warmup_s: float = 7.0
    resend_s: float = 0.2
    # Stay armed (lidar spinning) this long after the last motion, so a
    # follow-up doesn't pay the warm-up again.
    idle_disarm_s: float = 20.0
    # Measured: velocity 0.3 -> 12.6 cm/s; angle a -> ~a deg/s.
    cm_s_per_velocity: float = 42.0
    deg_s_per_angle: float = 1.0
    cruise_cm_s: float = 12.0          # the V1 ignores velocity >= 0.3 (12.6 cm/s)
    slow_cm_s: float = 5.0
    max_turn_rate_deg_s: float = 60.0
    min_turn_rate_deg_s: float = 15.0
    turn_gain: float = 1.5             # deg/s of rate per degree left to turn
    # It keeps going briefly after a stop: stop early by rate x coast_s.
    coast_s: float = 0.3
    settle_s: float = 0.6
    turn_tolerance_deg: float = 4.0
    move_tolerance_cm: float = 2.0
    max_turn_deg: float = 180.0
    max_move_cm: float = 100.0
    turn_timeout_s: float = 15.0
    # Leaving the dock on a low battery would only mean coming back.
    min_battery_to_leave: int = 40
    # Going home (spatial/dock.py): drive to a point this far straight out in
    # front of the dock first, unless already within dock_near_cm of it, then
    # dock from there. Not docked after dock_timeout_s: stop, go round again.
    dock_approach_cm: float = 60.0
    dock_near_cm: float = 30.0
    dock_timeout_s: float = 90.0
    dock_attempts: int = 2
    # A go_to ends "idle" whether it got there or not, so it arrived only if it
    # stopped this close. Measured 2026-09-26: 8-12 cm off when it arrives;
    # 25 cm off with the goal on furniture, 39 cm with it inside.
    arrive_cm: float = 20.0
    # Places are kept in this map's frame (spatial/frame.py). None yet: the
    # first map of at least reference_min_m2 becomes it.
    reference_map: str = "runtime/map/reference.json"
    reference_min_m2: float = 30.0
    # Share of the current map's walls that must land on the reference's.
    # Measured: 0.74-0.85 aligned, 0.06-0.19 not.
    frame_min_score: float = 0.5


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
    # A known person in view counts as talking to the pet (with recognition
    # off, anyone in view).
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
    journal_in_prompt: int = 5
    facts_in_prompt: int = 15


@dataclass
class LogConfig:
    """A log folder per run of petd (petd/log.py)."""
    enabled: bool = True
    dir: str = "runtime/logs"
    keep_runs: int = 30                       # newest run folders kept; older ones deleted
    max_old_runs_mb: float = 120.0            # ...and the older runs together at most this big
    file_level: str = "DEBUG"                 # the console keeps --log-level
    max_file_mb: float = 20.0                 # petd.log rotates at this size...
    keep_files: int = 3                       # ...keeping this many older ones
    events: bool = True                       # every bus event to events.jsonl (petd/trace.py),
    # which rotates at max_file_mb too, keeping one older file. With the
    # defaults a run is at most 80 + 40 MB, and runtime/logs ~240 MB in all.


@dataclass
class RecognitionConfig:
    """
    Face recognition on the PC (PLAN.md 4.7, E6; petd/vision/). The head only
    detects. Thresholds are SFace cosine similarities, set from E6a
    (runtime/e6a/: Robin and Claudia at 0.6-2.5 m, lamp light): the other
    person never scored above 0.29 against someone's centre, the person
    themselves never below 0.43.
    """
    enabled: bool = True                      # also needs the models: scripts/fetch_face_models.py
    models_dir: str = "models/face"
    detector_model: str = "face_detection_yunet_2023mar.onnx"
    recognizer_model: str = "face_recognition_sface_2021dec.onnx"
    # Gates: skip rather than guess. 45 px is ~2.5 m at VGA, where E6a still
    # told two people apart. No blur gate: sharpness varies too much with
    # person and distance to have one cut-off (E6a).
    min_face_px: float = 45.0
    min_detection_score: float = 0.8         # YuNet scored real faces 0.86-0.95
    min_brightness: float = 40.0             # mean of the aligned face, 0-255
    # One attempt: below unknown_sim, or within margin of the next person: unknown.
    unknown_sim: float = 0.35
    margin: float = 0.15
    # The vote over a visit's attempts (Frigate-style): a name needs min_agree
    # agreeing attempts, no tie, and a weighted mean of at least accept_sim.
    accept_sim: float = 0.45
    min_agree: int = 2
    # Attempts: one snapshot every attempt_every_s while someone in view is
    # unknown, at most max_attempts a visit (plus confirm_attempts once named).
    attempt_every_s: float = 1.5
    max_attempts: int = 12
    confirm_attempts: int = 6
    # While a named face is in view, look again this often (0: never), so a
    # wrong name doesn't last the whole conversation. ~2 snapshots each.
    recheck_every_s: float = 10.0
    # Growth: a confident attempt (at least grow_sim to the person it was voted
    # as) is kept as another fingerprint if it's a new look: less than
    # duplicate_sim to every one kept. At most grow_per_visit a visit; once a
    # person has max_per_person, a newer look replaces their most redundant
    # grown one. The first live visits kept every attempt: after its first look,
    # a visit's looks were 0.89-0.95 alike to one kept (same place, same light),
    # and 30 were full within three visits.
    grow_sim: float = 0.55
    duplicate_sim: float = 0.90
    grow_per_visit: int = 2
    max_per_person: int = 30
    # Enrollment: this many crops close up, then as many a step back.
    enroll_samples: int = 5
    enroll_timeout_s: float = 10.0
    enroll_step_back_s: float = 3.0          # after "take one step back", before the second set
    # The aligned crops behind attempts and fingerprints (petd/vision/kept.py).
    faces_dir: str = "runtime/faces"
    keep_attempts: int = 200


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
    motion: MotionConfig = field(default_factory=MotionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    log: LogConfig = field(default_factory=LogConfig)

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
