import asyncio
from collections import Counter
from collections.abc import Sequence

from langcheck.evaluation.models import ScoredEdit
from langcheck.judging.models import (
    EditJudgment,
    Item,
    Judged,
    JudgeResult,
    Judgment,
    NotJudged,
    SentenceJudgment,
    Source,
    Verdict,
)
from langcheck.judging.panel import merged, pool_item, run_panel, summarize, unjudged

GO = ScoredEdit(1, 2, ("went",), None)
THE = ScoredEdit(3, 3, ("the",), None)
TOKENS = ("He", "go", "to", "school", ".")


class FixedJudge:
    """Gives every edit one verdict; fails on sentences listed in ``fail_on``."""

    def __init__(self, name: str, verdict: Verdict, *, fail_on: Sequence[str] = ()) -> None:
        self._name = name
        self._verdict = verdict
        self._fail_on = set(fail_on)
        self.seen: list[Item] = []

    @property
    def name(self) -> str:
        return self._name

    async def judge(self, item: Item) -> JudgeResult:
        self.seen.append(item)
        if item.tokens[0] in self._fail_on:
            return NotJudged("no reply")
        verdicts = tuple((self._verdict, "r") for _ in item.candidates)
        return Judged(verdicts, self._verdict is not Verdict.VALID, "left")


class ListStore:
    def __init__(self, judgments: Sequence[Judgment] = ()) -> None:
        self.judgments = list(judgments)

    def load(self) -> list[Judgment]:
        return list(self.judgments)

    def add(self, judgments: Sequence[Judgment]) -> None:
        self.judgments += judgments


def test_pooling_merges_annotators_and_runs_and_remembers_where_edits_came_from() -> None:
    item = pool_item(TOKENS, gold=[(GO,), (GO, THE)], runs=[(THE,)])

    assert [(c.key, c.sources) for c in item.candidates] == [
        (GO.key, frozenset({Source.GOLD})),
        (THE.key, frozenset({Source.GOLD, Source.RUN})),
    ]


def test_a_repeated_sentence_is_one_item_with_every_edit() -> None:
    first = pool_item(TOKENS, gold=[(GO,)])
    again = pool_item(TOKENS, gold=[], runs=[(GO, THE)])

    [item] = merged([first, again])

    assert [(c.key, c.sources) for c in item.candidates] == [
        (GO.key, frozenset({Source.GOLD, Source.RUN})),
        (THE.key, frozenset({Source.RUN})),
    ]


def test_each_judge_stores_a_verdict_per_edit_and_on_the_sentence() -> None:
    store = ListStore()
    item = pool_item(TOKENS, gold=[(GO,)])

    runs = asyncio.run(run_panel([item, item], [FixedJudge("a", Verdict.VALID)], store))

    assert [(r.judge, r.judged, r.skipped, r.failures) for r in runs] == [("a", 1, 0, ())]
    assert store.judgments == [
        EditJudgment(
            sentence=item.key,
            start=1,
            end=2,
            correction=("went",),
            judge="a",
            verdict=Verdict.VALID,
            reason="r",
        ),
        SentenceJudgment(sentence=item.key, judge="a", remaining_error=False, reason="left"),
    ]


def test_a_later_run_judges_only_new_edits_and_failed_sentences() -> None:
    store = ListStore()
    first = pool_item(("A", "go"), gold=[(GO,)])
    second = pool_item(("B", "go"), gold=[(GO,)])
    asyncio.run(run_panel([first, second], [FixedJudge("a", Verdict.VALID, fail_on=["B"])], store))

    grown = pool_item(("A", "go"), gold=[(GO,)], runs=[(ScoredEdit(0, 1, ("An",), None),)])
    judge = FixedJudge("a", Verdict.VALID)
    [run] = asyncio.run(run_panel([grown, second, first], [judge], store))

    assert {item.tokens[0] for item in judge.seen} == {"A", "B"}
    assert [i.key for i in unjudged([first, second], "a", store.judgments)] == []
    assert (run.judged, run.skipped) == (2, 0)


def test_failures_are_reported_and_not_stored() -> None:
    store = ListStore()
    item = pool_item(("B", "go"), gold=[(GO,)])
    progress: list[str] = []

    [run] = asyncio.run(
        run_panel(
            [item], [FixedJudge("a", Verdict.VALID, fail_on=["B"])], store, progress=progress.append
        )
    )

    assert run.failures == ("no reply",)
    assert store.judgments == []
    assert progress == ["a: 1 of 1 sentences to judge", "a: 1/1 (1 failed)"]


def test_the_summary_counts_consensus_by_source_and_agreement_between_judges() -> None:
    store = ListStore()
    gold_only = pool_item(("A", "go"), gold=[(GO,)])
    shared = pool_item(("B", "go", "to", "x"), gold=[(GO,)], runs=[(THE,)])
    judges = [
        FixedJudge("a", Verdict.VALID),
        FixedJudge("b", Verdict.VALID),
        FixedJudge("c", Verdict.WRONG),
    ]
    asyncio.run(run_panel([gold_only, shared], judges, store))
    # A judge outside the panel never votes.
    store.add([SentenceJudgment(sentence=shared.key, judge="z", remaining_error=True, reason="")])

    summary = summarize([gold_only, shared], store.judgments, ["a", "b", "c"], quorum=2)

    assert summary.edits[Source.GOLD].agreed == Counter({"valid": 2})
    assert summary.edits[Source.RUN].agreed == Counter({"valid": 1})
    assert summary.remaining.agreed == Counter({"False": 2})
    assert summary.by_judge["c"] == Counter({Verdict.WRONG: 3})
    assert summary.agreement == {("a", "b"): 1.0, ("a", "c"): 0.0, ("b", "c"): 0.0}


def test_unjudged_edits_are_pending_and_split_votes_disputed() -> None:
    item = pool_item(("A", "go"), gold=[(GO,)])
    store = ListStore()
    asyncio.run(run_panel([item], [FixedJudge("a", Verdict.VALID)], store))

    pending = summarize([item], store.judgments, ["a", "b", "c"], quorum=2)
    asyncio.run(
        run_panel([item], [FixedJudge("b", Verdict.WRONG), FixedJudge("c", Verdict.NEUTRAL)], store)
    )
    disputed = summarize([item], store.judgments, ["a", "b", "c"], quorum=2)

    assert (pending.edits[Source.GOLD].pending, pending.edits[Source.GOLD].total) == (1, 1)
    assert disputed.edits[Source.GOLD].disputed == 1
    assert disputed.edits[Source.RUN].total == 0
