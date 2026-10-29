import asyncio

import pytest

from langcheck.check.protocols import OptionRanker, Ranking
from langcheck.check.suggestion import UNSCORED, Option, Scored, Suggestion
from langcheck.scoring.models import ScoredText
from langcheck.scoring.policy import KindPolicy, ScoringPolicy
from langcheck.scoring.ranker import KEEP_ORDER, KeepOrder, ScoringRanker
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Sentence, Span
from tests.conftest import TableScorer

SENTENCE = Sentence("S2", Span(10, 30), "I think their going.")
OTHER = Sentence("S3", Span(31, 40), "Its late.")


def suggestion(start: int, original: str, options: list[str], kind: ErrorKind) -> Suggestion:
    return Suggestion(
        id=f"sg_{start}",
        span=Span(start, start + len(original)),
        original=original,
        kind=kind,
        operation=Operation.REPLACEMENT,
        explanation="",
        options=tuple(Option(o) for o in options),
    )


THEIR = suggestion(18, "their going", ["they are going", "they're going"], ErrorKind.APOSTROPHE)
ITS = suggestion(31, "Its", ["It's"], ErrorKind.PUNCTUATION)
SCORES = {
    "I think their going.": -5.0,
    "I think they're going.": -2.0,
    "I think they are going.": -3.0,
    "Its late.": -1.0,
    "It's late.": -1.25,
}


def rank(ranker: OptionRanker, suggestions: list[Suggestion]) -> Ranking:
    return asyncio.run(ranker.rank(suggestions, [SENTENCE, OTHER]))


def test_keep_order_changes_nothing() -> None:
    ranker = KeepOrder()

    assert ranker.name == KEEP_ORDER
    assert rank(ranker, [THEIR]) == Ranking((THEIR,), ())


def test_options_are_rated_and_sorted_by_delta_in_one_batch() -> None:
    scorer = TableScorer(SCORES)
    ranker = ScoringRanker(scorer, policy=ScoringPolicy(), name="pll")

    ranking = rank(ranker, [THEIR, ITS])

    [their, its] = ranking.kept
    assert [(o.replacement, o.rating) for o in their.options] == [
        ("they're going", Scored(score=-2.0, delta=3.0)),
        ("they are going", Scored(score=-3.0, delta=2.0)),
    ]
    assert its.options == (Option("It's", Scored(score=-1.25, delta=-0.25)),)
    assert ranking.dropped == ()
    assert ranker.name == "pll"
    [batch] = scorer.batches
    assert batch[0] == ScoredText("I think their going.", Span(8, 19))
    assert batch[1] == ScoredText("I think they are going.", Span(8, 22))
    assert batch[3] == ScoredText("Its late.", Span(0, 3))


def test_policy_drops_filtered_kinds_that_no_option_improves() -> None:
    policy = ScoringPolicy({ErrorKind.PUNCTUATION: KindPolicy(filter=True)})
    ranker = ScoringRanker(TableScorer(SCORES), policy=policy, name="pll")

    ranking = rank(ranker, [THEIR, ITS])

    assert [s.original for s in ranking.kept] == ["their going"]
    assert [s.original for s in ranking.dropped] == ["Its"]


def test_equal_deltas_keep_the_llm_order() -> None:
    ranker = ScoringRanker(TableScorer({}), policy=ScoringPolicy(), name="flat")

    [ranked] = rank(ranker, [THEIR]).kept

    assert [o.replacement for o in ranked.options] == ["they are going", "they're going"]
    assert all(o.rating == Scored(0.0, 0.0) for o in ranked.options)
    assert THEIR.options[0].rating == UNSCORED


def test_nothing_to_rank_needs_no_scoring() -> None:
    scorer = TableScorer(SCORES)

    assert rank(ScoringRanker(scorer, policy=ScoringPolicy(), name="pll"), []) == Ranking((), ())
    assert scorer.batches == []


def test_a_suggestion_outside_the_sentences_is_an_error() -> None:
    stray = suggestion(100, "x", ["y"], ErrorKind.SPELLING)

    with pytest.raises(ValueError, match="none of the given sentences"):
        rank(ScoringRanker(TableScorer(SCORES), policy=ScoringPolicy(), name="pll"), [stray])
