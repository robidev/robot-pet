"""
Parses the LLM's streamed reply into speakable sentences and inline
actions, in order, as the text arrives.

The pet's expressive actions are inline tags rather than tool calls, so
they cost no extra round-trip and work with any model:

    [emote:curious] Oh, it's you. [look:left] Again.

-> Action("emote", "curious"), Sentence("Oh, it's you."),
   Action("look", "left"), Sentence("Again.")

Sentences are emitted as soon as they're complete so speech starts while
the model is still writing.

Angle-bracket tags are never spoken. The Claude CLI puts a
<system-reminder> (tokens left) into every message the model gets, and on
2026-10-24 the model began each reply with a copy of it, read aloud. A
whole <system-reminder>...</system-reminder> block is dropped, and so is
any other <tag> on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

TAG = re.compile(r"\[([a-z_]+)(?::([^\]]*))?\]", re.IGNORECASE)
# A sentence ends at . ! ? or … (plus any closing quote/bracket), followed
# by whitespace or the end of the buffer.
SENTENCE_END = re.compile(r'([.!?…]+["\'\)\]]*)(\s+|$)')
# The CLI's note to the model, echoed back: dropped with what's inside it.
# Lazy, so a block with a second opener inside ends at the first closer.
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.IGNORECASE | re.DOTALL)
REMINDER_OPEN = "<system-reminder"
ANGLE_TAG = re.compile(r"</?[a-z][\w:-]*>", re.IGNORECASE)
# Fallback for a model that runs on without punctuation.
MAX_SENTENCE_CHARS = 220


@dataclass(frozen=True)
class Sentence:
    text: str


@dataclass(frozen=True)
class Action:
    kind: str           # emote | look | nod | shake | pause (| rest, the brain's own)
    value: str = ""


Piece = Union[Sentence, Action]


def _sentence(text: str) -> Sentence:
    # Removing an inline tag leaves the spaces that surrounded it.
    return Sentence(re.sub(r"\s+", " ", text).strip())


class SpeechStreamParser:
    """Feed streamed text in, get sentences and actions out, in order."""

    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> list[Piece]:
        self._buffer += chunk
        return self._drain(final=False)

    def flush(self) -> list[Piece]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[Piece]:
        self._buffer = REMINDER.sub(" ", self._buffer)
        held = ""
        cut = self._unfinished_angle()
        if cut is not None:
            if final:
                # A reminder never closed: better silent than its contents spoken.
                if self._buffer.lower().startswith(REMINDER_OPEN, cut):
                    self._buffer = self._buffer[:cut]
            else:
                # Still arriving: hold it (and anything after it) back.
                self._buffer, held = self._buffer[:cut], self._buffer[cut:]
        self._buffer = ANGLE_TAG.sub(" ", self._buffer)
        pieces = self._drain_brackets(final)
        self._buffer += held
        return pieces

    def _unfinished_angle(self) -> int | None:
        """Where an unclosed reminder, or a '<' with no '>' yet, starts."""
        starts = []
        opener = self._buffer.lower().find(REMINDER_OPEN)
        if opener != -1:
            starts.append(opener)
        last = self._buffer.rfind("<")
        if last != -1 and ">" not in self._buffer[last:]:
            starts.append(last)
        return min(starts) if starts else None

    def _drain_brackets(self, final: bool) -> list[Piece]:
        pieces: list[Piece] = []
        while True:
            match = TAG.search(self._buffer)
            if match:
                # Text before the tag is split normally; an incomplete
                # sentence stays buffered, so a mid-sentence tag is applied
                # slightly ahead of the words around it rather than
                # chopping them into separate utterances.
                done, rest = self._split(self._buffer[:match.start()], final=False)
                pieces += done
                pieces.append(Action(match.group(1).lower(), (match.group(2) or "").strip().lower()))
                self._buffer = rest + self._buffer[match.end():]
                continue
            # A '[' with no closing ']' yet may be a tag still arriving:
            # hold it (and anything after it) until the rest turns up.
            open_bracket = self._buffer.rfind("[")
            if open_bracket != -1 and not final:
                done, rest = self._split(self._buffer[:open_bracket], final=False)
                pieces += done
                self._buffer = rest + self._buffer[open_bracket:]
                break
            done, self._buffer = self._split(self._buffer, final=final)
            pieces += done
            break
        return [p for p in pieces if not (isinstance(p, Sentence) and not p.text)]

    def _split(self, text: str, final: bool) -> tuple[list[Piece], str]:
        """Complete sentences out of `text`, plus whatever is left over."""
        pieces: list[Piece] = []
        rest = text
        while True:
            match = SENTENCE_END.search(rest)
            end = match.end(1) if match else len(rest)
            if end > MAX_SENTENCE_CHARS:
                # Too long to say in one breath: break at the last comma or
                # space before the limit (a model writing without punctuation).
                cut = max(rest.rfind(",", 0, MAX_SENTENCE_CHARS),
                          rest.rfind(" ", 0, MAX_SENTENCE_CHARS))
                if cut > 0:
                    pieces.append(_sentence(rest[:cut + 1]))
                    rest = rest[cut + 1:]
                    continue
            if not match:
                break
            pieces.append(_sentence(rest[:match.end(1)]))
            rest = rest[match.end():]
        if final and rest.strip():
            pieces.append(_sentence(rest))
            rest = ""
        return pieces, rest
