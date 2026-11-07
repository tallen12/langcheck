import asyncio
from collections import Counter
from collections.abc import Sequence

from langcheck.evaluation.models import ScoredEdit
from langcheck.judging.models import (
    Choice,
    EditJudgment,
    EditVerdict,
    Item,
    Judged,
    JudgeResult,
    Judgment,
    NotJudged,
    Preference,
    PreferenceJudgment,
    Preferred,
    SentenceJudgment,
    Source,
    Tied,
    Verdict,
    Version,
)
from langcheck.judging.panel import merged, pool_item, run_panel, summarize, unjudged

GO = ScoredEdit(1, 2, ("went",), None)
THE = ScoredEdit(3, 3, ("the",), None)
TOKENS = ("He", "go", "to", "school", ".")


class FixedJudge:
    """Gives every edit one verdict and links, prefers what it is told; fails on
    sentences whose first token is in ``fail_on``.
    """

    def __init__(
        self,
        name: str,
        verdict: Verdict,
        *,
        fail_on: Sequence[str] = (),
        link_all: bool = False,
        preference: Preference | None = None,
    ) -> None:
        self._name = name
        self._verdict = verdict
        self._fail_on = set(fail_on)
        self._link_all = link_all
        self._preference = preference
        self.seen: list[Item] = []

    @property
    def name(self) -> str:
        return self._name

    async def judge(self, item: Item) -> JudgeResult:
        self.seen.append(item)
        if item.tokens[0] in self._fail_on:
            return NotJudged("no reply")
        count = len(item.candidates)
        verdicts = tuple(
            EditVerdict(
                self._verdict,
                "r",
                tuple(j for j in range(count) if j != i) if self._link_all else (),
            )
            for i in range(count)
        )
        preference = self._preference if len(item.versions) > 1 else None
        return Judged(verdicts, self._verdict is not Verdict.VALID, "left", preference)


class ListStore:
    def __init__(self, judgments: Sequence[Judgment] = ()) -> None:
        self.judgments = list(judgments)

    def load(self) -> list[Judgment]:
        return list(self.judgments)

    def add(self, judgments: Sequence[Judgment]) -> None:
        self.judgments += judgments


def test_pooling_keeps_each_correction_and_where_its_edits_came_from() -> None:
    item = pool_item(TOKENS, gold=[(GO,), (GO, THE), (GO,), ()], runs=[("run", (THE,))])

    assert [(c.key, c.sources) for c in item.candidates] == [
        (GO.key, frozenset({Source.GOLD})),
        (THE.key, frozenset({Source.GOLD, Source.RUN})),
    ]
    assert item.versions == (
        Version("annotator 1, annotator 3", (GO.key,)),
        Version("annotator 2", (GO.key, THE.key)),
        Version("run", (THE.key,)),
    )


def test_a_repeated_sentence_is_one_item_with_every_correction() -> None:
    first = pool_item(TOKENS, gold=[(GO,)], written="He go to school.")
    again = pool_item(TOKENS, gold=[], runs=[("run", (GO, THE))])

    [item] = merged([first, again])

    assert item.written == "He go to school."
    assert [(c.key, c.sources) for c in item.candidates] == [
        (GO.key, frozenset({Source.GOLD, Source.RUN})),
        (THE.key, frozenset({Source.RUN})),
    ]
    assert [v.label for v in item.versions] == ["annotator 1", "run"]


def test_each_judge_stores_verdicts_links_the_sentence_and_its_preference() -> None:
    store = ListStore()
    item = pool_item(TOKENS, gold=[(GO, THE), (GO,)])
    judge = FixedJudge("a", Verdict.VALID, link_all=True, preference=Preferred(1, "simpler"))

    runs = asyncio.run(run_panel([item, item], [judge], store))

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
            linked=(THE.key,),
        ),
        EditJudgment(
            sentence=item.key,
            start=3,
            end=3,
            correction=("the",),
            judge="a",
            verdict=Verdict.VALID,
            reason="r",
            linked=(GO.key,),
        ),
        SentenceJudgment(sentence=item.key, judge="a", remaining_error=False, reason="left"),
        PreferenceJudgment(
            sentence=item.key,
            judge="a",
            versions=("annotator 1", "annotator 2"),
            choice=Choice.VERSION,
            preferred="annotator 2",
            reason="simpler",
        ),
    ]


def test_a_later_run_judges_only_new_edits_and_failed_sentences() -> None:
    store = ListStore()
    first = pool_item(("A", "go"), gold=[(GO,)])
    second = pool_item(("B", "go"), gold=[(GO,)])
    asyncio.run(run_panel([first, second], [FixedJudge("a", Verdict.VALID, fail_on=["B"])], store))

    grown = pool_item(("A", "go"), gold=[(GO,)], runs=[("run", (ScoredEdit(0, 1, ("An",), None),))])
    judge = FixedJudge("a", Verdict.VALID)
    [run] = asyncio.run(run_panel([grown, second, first], [judge], store))

    assert {item.tokens[0] for item in judge.seen} == {"A", "B"}
    assert [i.key for i in unjudged([first, second], "a", store.judgments)] == []
    assert (run.judged, run.skipped) == (2, 0)


def test_failures_are_reported_and_not_stored() -> None:
    store = ListStore()
    item = pool_item(("B", "go"), gold=[(GO,)])
    progress: list[str] = []
    judge = FixedJudge("a", Verdict.VALID, fail_on=["B"])

    [run] = asyncio.run(run_panel([item], [judge], store, progress=progress.append))

    assert run.failures == ("no reply",)
    assert store.judgments == []
    assert progress == ["a: 1 of 1 sentences to judge", "a: 1/1 (1 failed)"]


def test_the_summary_counts_consensus_preferences_and_agreement() -> None:
    store = ListStore()
    gold_only = pool_item(("A", "go"), gold=[(GO,)])
    shared = pool_item(("B", "go", "to", "x"), gold=[(GO,)], runs=[("run", (THE,))])
    judges = [
        FixedJudge("a", Verdict.VALID, preference=Preferred(0, "")),
        FixedJudge("b", Verdict.VALID, preference=Tied("")),
        FixedJudge("c", Verdict.WRONG, preference=Preferred(0, "")),
    ]
    asyncio.run(run_panel([gold_only, shared], judges, store))
    # A judge outside the panel never votes.
    store.add([SentenceJudgment(sentence=shared.key, judge="z", remaining_error=True, reason="")])

    summary = summarize([gold_only, shared], store.judgments, ["a", "b", "c"], quorum=2)

    assert summary.edits[Source.GOLD].agreed == Counter({"valid": 2})
    assert summary.edits[Source.RUN].agreed == Counter({"valid": 1})
    assert summary.linked == {Source.GOLD: 0, Source.RUN: 0}
    assert summary.preferences.agreed == Counter({"annotator 1": 1})
    assert summary.remaining.agreed == Counter({"False": 2})
    assert summary.by_judge["c"] == Counter({Verdict.WRONG: 3})
    assert summary.agreement == {("a", "b"): 1.0, ("a", "c"): 0.0, ("b", "c"): 0.0}


def test_linked_edits_count_by_consensus_or_by_shape() -> None:
    pair = pool_item(TOKENS, gold=[(GO, THE)])
    move = pool_item(
        ("We", "see", "it", "often", "."),
        gold=[(ScoredEdit(3, 4, (), None), ScoredEdit(1, 1, ("often",), None))],
    )
    store = ListStore()
    judges = [FixedJudge(name, Verdict.VALID, link_all=name != "c") for name in "abc"]
    asyncio.run(run_panel([pair], judges, store))
    asyncio.run(run_panel([move], [FixedJudge(n, Verdict.VALID) for n in "abc"], store))

    summary = summarize([pair, move], store.judgments, ["a", "b", "c"], quorum=2)

    assert summary.linked[Source.GOLD] == 4


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
    assert disputed.preferences.total == 0
