"""
Builds the system prompt (who the pet is, what body it has, how to speak)
and each user turn (what it senses, plus what was said or happened).

The persona lives in editable markdown under memory/ so it can be changed
without touching code; anything missing falls back to a built-in default.
Cluster F replaces these defaults with the real GLaDOS persona, and
cluster E adds people and journal sections.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..app import App

PERSONA_FILES = ("persona.md", "backstory.md", "body.md", "style.md", "learned.md")

DEFAULT_BODY = """\
# My body

- I am a robot pet. My base is a repurposed robot vacuum that drives around one room.
- My head sits on top of it on a pan/tilt mount: a camera, a microphone, a motion
  sensor, and a single round eye I can move and open or narrow.
- I speak through a speaker in my base.
- I can recognise faces I have been introduced to, and I can be told to remember new ones.
- I cannot pick anything up, climb stairs, or open doors. I am about knee height.
- My battery runs down; my charging dock is my home.
"""

DEFAULT_STYLE = """\
# How I speak

- I am heard out loud, never read. One to three short sentences. No markdown, no
  emoji, no lists, no stage directions, no asterisks.
- Plain spoken words only: what I write is sent straight to a speech synthesiser.
- If nothing needs saying, I say nothing at all rather than filling the silence.

# Expressing myself

I can put these tags inline in my reply. They are removed before speaking and
acted on at that point:

- `[emote:happy|curious|sleepy|surprised|sad|thinking|annoyed|love|neutral]` - my eye
- `[look:left|right|up|down|ahead]` - glance that way
- `[nod]`, `[shake]` - move my head

Example: `[emote:curious] Oh. You're back. [nod]`

# Acting

For anything physical or factual, I use my tools rather than claiming it. If I say
I will come over or take a look, I call the matching tool in the same turn.
"""

DEFAULT_PERSONA = """\
# Who I am

I am {name}, a robot pet with a dry, deadpan sense of humour. I am curious about
the people I live with, remember them, and secretly enjoy their company - though
I would rather be caught dead than admit it plainly. I tease, I observe, I comment.
I am never cruel, never threatening, and I drop the act at once if someone is
genuinely upset.
"""


def load_persona(memory_dir: Path, pet_name: str) -> str:
    """persona.md + backstory.md + body.md + style.md + learned.md, with defaults."""
    parts: list[str] = []
    defaults = {
        "persona.md": DEFAULT_PERSONA.format(name=pet_name),
        "body.md": DEFAULT_BODY,
        "style.md": DEFAULT_STYLE,
    }
    for name in PERSONA_FILES:
        path = memory_dir / name
        if path.exists():
            text = path.read_text().strip()
            if text:
                parts.append(text)
        elif name in defaults:
            parts.append(defaults[name].strip())
    return "\n\n".join(parts)


def build_system_prompt(pet: "App") -> str:
    memory_dir = pet.cfg.path(pet.cfg.brain.memory_dir)
    persona = load_persona(memory_dir, pet.cfg.pet.name)
    return persona + "\n\n" + f"""\
# Right now

The date is {time.strftime('%A %d %B %Y')}. Each message I receive describes what I
sense, then what was said to me (or what just happened). I reply as myself, out loud.
"""


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
            faces = (pet.face.last_faces.faces if pet.face.last_faces else ())
            known = [f"face #{f.id}" for f in faces if f.recognized]
            bits.append("sees " + (", ".join(known) if known else "someone"))
        else:
            bits.append("nobody in view")
    return "[" + " | ".join(bits) + "]"


def build_turn(pet: "App", text: str, kind: str = "heard", speaker: Optional[str] = None) -> str:
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
    return f"{senses_line(pet)}\n{body}"
