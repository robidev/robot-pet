"""
Typed events carried on the EventBus. Every event has `t`, the PC wall
clock (Unix epoch seconds) at creation. Device/sensor timestamps, where
they exist, are separate fields.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, kw_only=True)
class Event:
    t: float = field(default_factory=time.time)


# --- vacuum ------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class VacuumStateChanged(Event):
    state: Any                  # io.vacuum.VacuumState
    changed: tuple = ()         # names of the fields that changed


# --- face --------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class FacesChanged(Event):
    """Raw /ws face event: the set of detected faces changed."""
    faces: tuple                # tuple[io.face.Face, ...]
    pan_deg: float
    tilt_deg: float
    device_utc: Optional[float] # capture time per the device clock, epoch s
    seq: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class FacesPresence(Event):
    """Debounced: someone is (or no longer is) in view."""
    present: bool
    faces: tuple = ()


@dataclass(frozen=True, kw_only=True)
class MotionDetected(Event):
    active: bool
    pan_deg: float
    tilt_deg: float
    device_utc: Optional[float]


@dataclass(frozen=True, kw_only=True)
class FaceDeviceConnection(Event):
    connected: bool
    detail: str = ""


# --- hearing -----------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class SpeechStarted(Event):
    t_utc: float


@dataclass(frozen=True, kw_only=True)
class SpeechEnded(Event):
    t_utc: float
    discarded: bool = False


@dataclass(frozen=True, kw_only=True)
class Heard(Event):
    text: str
    t_start: float
    t_end: float
    no_speech_prob: float = 0.0
    source: str = "mic"         # mic | api | fake


@dataclass(frozen=True, kw_only=True)
class HeardDropped(Event):
    """A transcript that was filtered out, kept for debugging/dashboard."""
    text: str
    reason: str


# --- speaking ----------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class SpeakingStarted(Event):
    utterance_id: int


@dataclass(frozen=True, kw_only=True)
class SpeakingFinished(Event):
    utterance_id: int
    text: str
    interrupted: bool = False


# --- infrastructure ------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class ProcessStateChanged(Event):
    name: str
    running: bool
    returncode: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class StopRequested(Event):
    source: str
