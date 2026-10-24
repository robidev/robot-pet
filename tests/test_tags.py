from petd.brain.tags import Action, Sentence, SpeechStreamParser


def parse(chunks):
    p = SpeechStreamParser()
    out = []
    for chunk in chunks:
        out += p.feed(chunk)
    return out + p.flush()


def test_sentences_stream_out_as_they_complete():
    p = SpeechStreamParser()
    assert p.feed("Oh. ") == [Sentence("Oh.")]
    assert p.feed("You again") == []                 # incomplete: held back
    # A sentence ending exactly at a chunk boundary goes out immediately:
    # speech should start without waiting for the next delta.
    assert p.feed(". Wonderful!") == [Sentence("You again."), Sentence("Wonderful!")]
    assert p.flush() == []


def test_tags_become_actions_in_order():
    assert parse(["[emote:curious] Oh. [look:left] You again."]) == [
        Action("emote", "curious"), Sentence("Oh."),
        Action("look", "left"), Sentence("You again."),
    ]


def test_tag_split_across_chunks_is_not_spoken():
    # The delta boundary can fall anywhere, including inside a tag.
    assert parse(["[emote:cur", "ious] Oh.", " You again."]) == [
        Action("emote", "curious"), Sentence("Oh."), Sentence("You again."),
    ]
    assert parse(["Hello", " [", "nod", "] there."]) == [
        Action("nod", ""), Sentence("Hello there."),
    ]


def test_bare_tags_and_unknown_tags():
    assert parse(["[nod][shake]Fine."]) == [
        Action("nod", ""), Action("shake", ""), Sentence("Fine."),
    ]
    assert parse(["[wiggle:3] Hm."]) == [Action("wiggle", "3"), Sentence("Hm.")]


def test_long_run_on_text_is_broken_up():
    text = "and then " * 40 + "the end."
    pieces = parse([text])
    assert all(isinstance(p, Sentence) for p in pieces)
    assert len(pieces) > 1
    assert max(len(p.text) for p in pieces) <= 221
    assert "".join(p.text for p in pieces).endswith("the end.")


def test_quotes_and_ellipses_end_sentences():
    assert parse(['She said "no." Then... nothing.']) == [
        Sentence('She said "no."'), Sentence("Then..."), Sentence("nothing."),
    ]


def test_system_reminder_echo_is_not_spoken():
    # As the model wrote it on 2026-10-24, delta by delta (runtime/logs/20261024-175439).
    assert parse(["<system", "-reminder>\n<total", "_tokens>14", "999686", " tokens left</tml",
                  ":total", "_tokens>\n</system", "-reminder>[", "em", "ote:neutral]"]) == [
        Action("emote", "neutral"),
    ]
    assert parse(["<system-reminder>\n<total_tokens>14999675 tokens left</total_tokens>\n"
                  "</system-reminder>\n\n[emote:curious] [pause] I'm still listening."]) == [
        Action("emote", "curious"), Action("pause", ""), Sentence("I'm still listening."),
    ]
    # Two openers, one closer.
    assert parse(["<system-reminder> <total_tokens>1 tokens left</total_tokens><system-reminder>",
                  " <total_tokens>2 tokens left</total_tokens> </system-reminder> I don't control those."]) == [
        Sentence("I don't control those."),
    ]


def test_speech_before_a_reminder_still_streams_out():
    p = SpeechStreamParser()
    assert p.feed("Oh. <sys") == [Sentence("Oh.")]
    assert p.feed("tem-reminder>12 tokens") == []
    assert p.feed(" left</system-reminder> Hi.") == [Sentence("Hi.")]
    assert p.flush() == []


def test_unclosed_reminder_and_stray_tags():
    assert parse(["Fine. <system-reminder> <total_tokens>3 tokens left"]) == [Sentence("Fine.")]
    assert parse(["<total_tokens>Hello</total_tokens> there."]) == [Sentence("Hello there.")]
    assert parse(["Three < five."]) == [Sentence("Three < five.")]
