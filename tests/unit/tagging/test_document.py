from collections.abc import Sequence

from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import flag_text, sentence_spans
from langcheck.tagging.models import SlotScores
from langcheck.tagging.tiers import FlagKind, Tier, TierPolicy


def pieces(text: str) -> list[str]:
    return [text[span.start : span.end] for span in sentence_spans(text)]


def test_text_splits_after_final_punctuation_and_a_space() -> None:
    assert pieces("He ran.  Did she? Yes! ") == ["He ran.", "Did she?", "Yes!"]


def test_closing_quotes_stay_with_their_sentence() -> None:
    assert pieces('He said "go." She went.') == ['He said "go."', "She went."]


def test_line_breaks_end_sentences() -> None:
    assert pieces("A title\n\nThe text") == ["A title", "The text"]


def test_abbreviations_initials_and_numbers_do_not_end_sentences() -> None:
    assert pieces("Dr. Smith paid 3.50 to J. Lee, e.g. cash. He left.") == [
        "Dr. Smith paid 3.50 to J. Lee, e.g. cash.",
        "He left.",
    ]


def test_a_sentence_without_final_punctuation_is_kept() -> None:
    assert pieces("He ran. she left") == ["He ran.", "she left"]


def test_empty_text_has_no_sentences() -> None:
    assert sentence_spans("  \n ") == []


class FlagsWord:
    """Flags the word ``go`` in every sentence and records what it was given."""

    def __init__(self) -> None:
        self.sentences: list[Sequence[str]] = []
        self.contexts: Sequence[str] | None = None

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        self.sentences = list(sentences)
        self.contexts = contexts
        return [
            SlotScores(
                tuple(0.9 if word == "go" else 0.0 for word in s),
                (0.0,) * (len(s) + 1),
            )
            for s in sentences
        ]


class NoCommas:
    def __init__(self) -> None:
        self.contexts: Sequence[str] | None = None

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[tuple[float, ...]]:
        self.contexts = contexts
        return [(0.0,) * (len(s) + 1) for s in sentences]


TEXT = "I ran home.  We go  there. They go too."


def test_flags_point_into_the_whole_text() -> None:
    flags = flag_text(TEXT, FlagsWord(), None, policy=TierPolicy(), context_sentences=2)

    assert [TEXT[f.span.start : f.span.end] for f in flags] == ["go", "go"]
    assert flags[0].span == CharSpan(16, 18)
    assert flags[0].tier is Tier.ERROR
    assert flags[0].kind is FlagKind.GENERAL


def test_each_sentence_reads_the_ones_before_it() -> None:
    scorer, commas = FlagsWord(), NoCommas()

    flag_text(TEXT, scorer, commas, policy=TierPolicy(), context_sentences=1)

    assert scorer.sentences[1] == ("We", "go", "there", ".")
    assert scorer.contexts == ["", "I ran home.", "We go there."]
    assert commas.contexts == scorer.contexts


def test_a_tagger_without_context_reads_none() -> None:
    scorer = FlagsWord()

    flag_text(TEXT, scorer, None, policy=TierPolicy(), context_sentences=0)

    assert scorer.contexts == ["", "", ""]


def test_text_without_words_is_not_scored() -> None:
    scorer = FlagsWord()

    assert flag_text("  ", scorer, None, policy=TierPolicy(), context_sentences=2) == []
    assert scorer.sentences == []
