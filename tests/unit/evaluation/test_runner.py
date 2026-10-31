import asyncio
from collections import Counter
from collections.abc import Collection

from langcheck.check.checker import CheckResult, Unchecked
from langcheck.check.reply import ReplyIssue
from langcheck.check.resolve import Rejection, RejectionReason
from langcheck.check.suggestion import Option, Scored, Suggestion, suggestion_id
from langcheck.context.models import DocumentContext
from langcheck.evaluation.closeness import Closeness
from langcheck.evaluation.examples import Example
from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.runner import EVALUATION_USER, Effect, apply_best, check_all, evaluate
from langcheck.evaluation.scoring import Counts
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span
from tests.conftest import TokenDiffAnnotator

GO = Example(
    ("He", "go", "teh", "way", "."),
    ((ScoredEdit(1, 2, ("goes",), None), ScoredEdit(2, 3, ("the",), None)),),
    "He go teh way .",
)
FINE = Example(("All", "fine", "."), ((),), "All fine .")


def suggestion(
    text: str,
    original: str,
    replacement: str,
    kind: ErrorKind,
    delta: float | None,
    *others: str,
) -> Suggestion:
    start = text.index(original)
    span = Span(start, start + len(original))
    rating = Scored(score=0.0, delta=delta) if delta is not None else Option(replacement).rating
    return Suggestion(
        id=suggestion_id(span, original, kind),
        span=span,
        original=original,
        kind=kind,
        operation=Operation.REPLACEMENT,
        explanation="",
        options=(Option(replacement, rating), *(Option(other) for other in others)),
    )


class ScriptedChecker:
    """Fixes "go" and "teh" (wrongly, as "a"; "the" is its third option), flags "fine",
    records calls.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, DocumentContext]] = []

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        self.calls.append((text, user, context))
        if "go" in text.split():
            return CheckResult(
                (
                    suggestion(text, "go", "goes", ErrorKind.SUBJECT_VERB_AGREEMENT, 2.0),
                    suggestion(text, "teh", "a", ErrorKind.SPELLING, None, "tea", "the"),
                ),
                (),
                (
                    Rejection(
                        ReplyIssue(
                            sentence="S1", original="x", kind="Style", explanation="", options=["y"]
                        ),
                        RejectionReason.UNKNOWN_KIND,
                    ),
                ),
                (),
            )
        return CheckResult(
            (suggestion(text, "fine", "good", ErrorKind.REAL_WORD, -0.5),),
            (Unchecked(Span(0, 3), "LLM error"),),
            (),
            (),
        )


def test_evaluate_scores_sentences_and_suggestions() -> None:
    checker = ScriptedChecker()

    run = asyncio.run(evaluate([GO, FINE], checker, TokenDiffAnnotator(), concurrency=2))

    assert run.score.overall == Counts(tp=1, fp=2, fn=1)
    assert run.top_n == (
        Counts(tp=1, fp=2, fn=1),
        Counts(tp=1, fp=2, fn=1),
        Counts(tp=2, fp=1, fn=0),
        Counts(tp=2, fp=1, fn=0),
    )
    go, fine = run.sentences
    assert [o.gain for o in go.outcomes] == [1, 0]
    assert [o.gain for o in fine.outcomes] == [-1]
    assert (go.closeness, fine.closeness) == (Closeness(2, 1), Closeness(0, -1))
    assert run.closeness == Closeness(2, 0)
    assert go.tokenized
    assert go.top_n[0] == go.score.overall
    assert [(o.kind, o.delta, o.effect) for o in go.outcomes] == [
        (ErrorKind.SUBJECT_VERB_AGREEMENT, 2.0, Effect.CORRECT),
        (ErrorKind.SPELLING, None, Effect.WRONG),
    ]
    assert [(o.kind, o.effect) for o in fine.outcomes] == [(ErrorKind.REAL_WORD, Effect.WRONG)]
    assert run.rejections == Counter({RejectionReason.UNKNOWN_KIND: 1})
    assert (go.unchecked, fine.unchecked) == (False, True)
    assert {(text, user) for text, user, _ in checker.calls} == {
        ("He go teh way .", EVALUATION_USER),
        ("All fine .", EVALUATION_USER),
    }
    assert all(context == DocumentContext.empty() for _, _, context in checker.calls)


def test_check_all_times_each_check() -> None:
    ticks = iter([0.0, 1.5, 2.0, 2.25])

    checks = asyncio.run(
        check_all([GO, FINE], ScriptedChecker(), concurrency=1, clock=lambda: next(ticks))
    )

    assert [seconds for _, seconds in checks] == [1.5, 0.25]


def test_apply_best_replaces_spans_in_order() -> None:
    text = "He go teh way ."
    later = suggestion(text, "teh", "the", ErrorKind.SPELLING, None)
    earlier = suggestion(text, "go", "goes", ErrorKind.SUBJECT_VERB_AGREEMENT, None)

    assert apply_best(text, [later, earlier]) == "He goes the way ."
    assert apply_best(text, []) == text
