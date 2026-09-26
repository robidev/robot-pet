"""
Builds the system prompt (who the pet is, what body it has, how to speak)
and each user turn (what it senses, plus what was said or happened).

The persona lives in editable markdown under memory/ so it can be changed
without touching code; a missing file is logged and left out.
The people I know, my journal and what I've learned come from the
memory database (petd/memory/) and are rebuilt at every episode start.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)

PERSONA_FILES = ("persona.md", "backstory.md", "body.md", "style.md")


def load_persona(memory_dir: Path) -> str:
    """persona.md + backstory.md + body.md + style.md; a missing one is logged and skipped."""
    parts: list[str] = []
    for name in PERSONA_FILES:
        path = memory_dir / name
        text = path.read_text().strip() if path.exists() else ""
        if text:
            parts.append(text)
        else:
            log.warning("persona file missing or empty: %s", path)
    return "\n\n".join(parts)


def memory_sections(pet: "App") -> str:
    """People I know, my journal and things I've learned, from the database."""
    db = pet.db
    if db is None:
        return ""
    from ..memory.people import FAMILIARITY_WORDS, ago
    cfg = pet.cfg.memory
    now = time.time()
    parts: list[str] = []

    people = db.people()
    lines = ["# People I know", ""]
    if not people:
        lines.append("Nobody yet. I have not been introduced to anyone.")
    for person in people:
        line = f"- {person.name}"
        if person.nickname:
            line += f" (I call them {person.nickname})"
        details = []
        if person.familiarity > 0:
            details.append(FAMILIARITY_WORDS[min(person.familiarity, 3)])
        details.append("I know their face" if db.face_count(person.id)
                       else "face not stored")
        line += ": " + ", ".join(details)
        if person.last_seen_at is not None:
            line += f", last seen {ago(now - person.last_seen_at)} ago"
        line += "."
        if person.notes:
            line += f" {person.notes}"
        # Closer people get more of what I know about them up front.
        facts = db.facts(about=person.id, limit=3 if person.familiarity >= 2 else 1)
        if facts:
            line += " " + " ".join(f.text.rstrip(".") + "." for f in facts)
        lines.append(line)
    parts.append("\n".join(lines))

    journal = db.journal(limit=cfg.journal_in_prompt)
    if journal:
        lines = ["# My journal (recent conversations)", ""]
        lines += [f"- {time.strftime('%a %d %b %H:%M', time.localtime(c.started_at))}: {c.summary}"
                  for c in journal]
        parts.append("\n".join(lines))

    facts = db.facts(limit=cfg.facts_in_prompt)
    if facts:
        lines = ["# Things I have learned", ""]
        lines += [f"- {f.text}" for f in reversed(facts)]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# Overrides what body.md says about faces while recognition is switched off.
RECOGNITION_OFF = """
My face recognition isn't running right now. I can see that someone is there,
but not who. So I never claim to recognise anyone, from their face or from a
photo, and I can't learn new faces. If it matters who I'm
talking to, I ask. A name someone gives me, I can still remember.
"""


def build_system_prompt(pet: "App") -> str:
    memory_dir = pet.cfg.path(pet.cfg.brain.memory_dir)
    persona = load_persona(memory_dir)
    memory = memory_sections(pet)
    if memory:
        persona += "\n\n" + memory
    now = f"""\
# Right now

The date is {time.strftime('%A %d %B %Y')}. Each message I receive describes what I
sense, then what was said to me (or what just happened). I reply as myself, out loud.
"""
    if not pet.recognition_on:
        now += RECOGNITION_OFF
    return persona + "\n\n" + now


def senses_line(pet: "App") -> str:
    """
    One compact line of state prefixed to a turn, e.g.
    [21:42 | battery 64% | docked | sees: someone straight ahead]
    """
    bits = [time.strftime("%H:%M")]
    if pet.vacuum is not None:
        state = pet.vacuum.state
        if not state.reachable:
            bits.append("body offline")
        else:
            if state.battery_level is not None:
                bits.append(f"battery {state.battery_level}%")
            if state.status:
                bits.append(state.status)
    if pet.face is not None:
        if not pet.face.state.reachable:
            bits.append("head offline")
        elif pet.face.presence.present:
            bits.append("sees " + _who_is_here(pet))
        else:
            bits.append("nobody in view")
    return "[" + " | ".join(bits) + "]"


def _who_is_here(pet: "App") -> str:
    if not pet.recognition_on:
        # Every face is unknown without recognition; that says nothing about who it is.
        faces = pet.face.last_faces.faces if pet.face.last_faces else ()
        return f"{len(faces)} people" if len(faces) > 1 else "someone"
    if pet.people is not None:
        names, strangers = pet.people.who_is_here()
    else:
        faces = pet.face.last_faces.faces if pet.face.last_faces else ()
        names = [f"face #{f.id}" for f in faces if f.recognized]
        strangers = len(faces) - len(names)
    if strangers:
        names.append("someone I don't recognize" if strangers == 1
                     else f"{strangers} people I don't recognize")
    return " and ".join(names) if names else "someone"


def build_turn(pet: "App", text: str, kind: str = "heard", speaker: Optional[str] = None,
               notes: Optional[list[str]] = None) -> str:
    """
    A user turn: the senses line plus what happened.

    kind="heard"  -> someone spoke to me
    kind="event"  -> something happened (arrival, a behavior finished, a timer);
                     the pet may answer with nothing at all.
    """
    who = speaker or "someone"
    if kind == "heard":
        body = f'{who} says: "{text}"'
    else:
        body = f"[event] {text}\n(Say something only if it's worth saying out loud.)"
    meanwhile = "".join(f"[meanwhile] {note}\n" for note in notes or ())
    return f"{senses_line(pet)}\n{meanwhile}{body}"


JOURNAL_REQUEST = (
    "[journal] This conversation is over. Write one or two plain sentences for your private "
    "journal: who you talked with, what about, and anything worth remembering next time. "
    "This is not spoken aloud. No tags, no tool calls."
)


def clean_summary(text: str) -> str:
    """The journal line as stored: tags stripped, whitespace collapsed, length capped."""
    import re
    text = re.sub(r"\[[a-z]+(?::[^\]]*)?\]", "", text)
    text = " ".join(text.split())
    return text[:400]
