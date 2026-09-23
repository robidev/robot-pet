"""Turning events and state into JSON: for the API, tool results and the event trace."""

from __future__ import annotations

import dataclasses
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj


def event_record(event: Any) -> dict:
    """An event as one JSON object, its class name under "type"."""
    return {"type": type(event).__name__, **to_jsonable(event)}
