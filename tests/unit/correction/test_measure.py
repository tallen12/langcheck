import asyncio
import json
from collections.abc import Callable, Sequence

from langcheck.correction.measure import (
    OFFERED,
    CorrectionReport,
    Strategy,
    StrategyRow,
    apply_all,
    measure,
    render_measurement,
    suggest_all,
)
from langcheck.correction.models import (
    Candidate,
    Confidence,
    Corrected,
    CorrectionPolicy,
    FlagOutcome,
    Ranked,
    Rejected,
    RejectionReason,
    Unavailable,
)
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.evaluation.examples import Example
from langcheck.evaluation.models import ScoredEdit
from langcheck.llm.models import ChatOptions, LLMError
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier, TierPolicy
from langcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import Responder, ScriptedLLM, TokenDiffAnnotator, WordTableTagger

SOURCE = ("He", "go", "home", ".")
GOLD = ScoredEdit(1, 2, ("went",), ErrorKind.VERB_TENSE)
EXAMPLE = Example(SOURCE, ((GOLD,),), "He go home .")
FLAG = TextFlag(CharSpan(3, 5), 0.9, Tier.ERROR, FlagKind.GENERAL)


def fix(replacement: str, position: int) -> Candidate:
    return Candidate(
        CharSpan(3, 5), "go", replacement, ErrorKind.VERB_TENSE, Operation.REPLACEMENT, "", position
    )


def ranked(replacement: str, position: int, after: float) -> Ranked:
    return Ranked(fix(replacement, position), 0.9, after, Confidence.HIGH)


def refused(replacement: str, position: int) -> Rejected:
    return Rejected(
        "go", replacement, RejectionReason.STILL_FLAGGED, 0.01, fix(replacement, position)
    )


def report(*outcomes: FlagOutcome) -> CorrectionReport:
    return measure([EXAMPLE], [list(outcomes)], TokenDiffAnnotator(), dataset="set")


def row(result: CorrectionReport, strategy: Strategy) -> StrategyRow:
    return next(r for r in result.strategies if r.strategy is strategy)


def test_apply_all_applies_in_text_order_and_skips_overlaps() -> None:
    first = Candidate(CharSpan(0, 2), "He", "She", ErrorKind.SPELLING, Operation.REPLACEMENT, "")
    second = fix("went", 0)
    overlap = fix("goes", 1)

    assert apply_all("He go home .", [second, overlap, first]) in {
        "She went home .",
        "She goes home .",
    }


def test_strategies_choose_differently() -> None:
    # The LLM likes "gone", the filters drop it, the tagger prefers "goes", "went" is right.
    outcome = Corrected(
        FLAG,
        (ranked("goes", 1, 0.0), ranked("went", 2, 0.1)),
        (refused("gone", 0),),
    )

    result = report(outcome)

    assert row(result, Strategy.LLM_FIRST).tp == 0
    assert row(result, Strategy.FILTERED_FIRST).tp == 0
    assert row(result, Strategy.TAGGER_RANKED).tp == 0
    best = row(result, Strategy.BEST_OFFERED)
    assert (best.tp, best.fp, best.fn) == (1, 0, 0)
    assert best.share == 1.0


def test_fates_show_what_each_filter_threw_away() -> None:
    outcome = Corrected(FLAG, (ranked("goes", 1, 0.0),), (refused("went", 0),))

    fates = {row.fate: row for row in report(outcome).fates}

    assert fates[RejectionReason.STILL_FLAGGED.value].helps == 1
    assert fates[OFFERED].helps == 0


def test_a_rejection_that_cannot_be_applied_is_left_out_of_the_records() -> None:
    outcome = Corrected(FLAG, (), (Rejected("x", "y", RejectionReason.NOT_AT_FLAG),))

    assert report(outcome).records == []


def test_order_counts_right_and_wrong_by_margin() -> None:
    near_tie = Corrected(FLAG, (ranked("goes", 0, 0.0), ranked("went", 1, 0.01)), ())
    clear = Corrected(FLAG, (ranked("went", 0, 0.0), ranked("goes", 1, 0.5)), ())

    close, wide = report(near_tie, clear).order

    assert (close.right, close.wrong) == (0, 1)
    assert (wide.right, wide.wrong) == (1, 0)


def test_flags_without_an_answer_or_corrections_are_counted() -> None:
    result = report(Unavailable(FLAG, "down"), Corrected(FLAG, (), ()))

    assert (result.flags, result.unavailable, result.no_corrections) == (2, 1, 1)


def test_doing_nothing_finds_nothing() -> None:
    best = row(report(Corrected(FLAG, (), ())), Strategy.TAGGER_RANKED)

    assert (best.tp, best.fp, best.fn, best.gain) == (0, 0, 1, 0)


def test_render_shows_every_section() -> None:
    text = render_measurement(report(Corrected(FLAG, (ranked("went", 0, 0.0),), ())))

    assert "tagger-ranked" in text
    assert f"{OFFERED}\t1\t1\t0\t0" in text
    assert "better first" in text


def test_suggest_all_keeps_order_and_logs_progress(
    scripted_llm: Callable[[Sequence[str | LLMError | Responder]], ScriptedLLM],
) -> None:
    pipeline = CorrectionPipeline(
        llm=scripted_llm([json.dumps({"flags": []})]),
        scorer=WordTableTagger({"go": 0.9}),
        commas=None,
        tiers=TierPolicy(),
        policy=CorrectionPolicy(),
        options=ChatOptions(),
        context_sentences=0,
    )
    logged: list[str] = []

    outcomes = asyncio.run(
        suggest_all(pipeline, ["He go.", "Fine."] * 13, concurrency=2, log=logged.append)
    )

    assert [len(found) for found in outcomes[:2]] == [1, 0]
    assert logged == ["25/26 sentences"]
