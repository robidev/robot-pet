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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

TAG = re.compile(r"\[([a-z_]+)(?::([^\]]*))?\]", re.IGNORECASE)
# A sentence ends at . ! ? or … (plus any closing quote/bracket), followed
# by whitespace or the end of the buffer.
SENTENCE_END = re.compile(r'([.!?…]+["\'\)\]]*)(\s+|$)')
# Fallback for a model that runs on without punctuation.
MAX_SENTENCE_CHARS = 220


@dataclass(frozen=True)
class Sentence:
    text: str


@dataclass(frozen=True)
class Action:
    kind: str           # emote | look | nod | shake | pause
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
