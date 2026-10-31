from collections.abc import Sequence

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.proposal.measure import (
    Case,
    build_cases,
    contexts_of,
    flagged_spans,
    gold_spans,
    normal,
    ranked,
    recall_row,
    tune,
)
from langcheck.proposal.protocols import Proposal, ProposalRequest
from langcheck.proposal.rerank import Scored
from langcheck.proposal.targets import WordSpan
from langcheck.tagging.models import SentenceReading, SlotScores
from langcheck.tagging.tiers import TierPolicy

SENTENCE = AnnotatedSentence(("He", "go", "home", "."), ((GoldEdit(1, 2, ("goes",), "R:VERB"),),))


def reading(words: Sequence[str]) -> SentenceReading:
    """Scores ``go`` as wrong, and nothing else."""
    return SentenceReading(
        SlotScores(
            tuple(0.9 if word == "go" else 0.0 for word in words), (0.0,) * (len(words) + 1)
        ),
        None,
    )


class Reader:
    def read(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SentenceReading]:
        return [reading(s) for s in sentences]


class Source:
    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        answer = (
            Proposal("", -0.1, keep=True),
            Proposal("go", -0.5),
            Proposal("", -0.7),
            Proposal("goes", -1.0),
        )
        return [[answer for _ in request.spans] for request in requests]


def case(*proposals: Scored, answers: frozenset[str] = frozenset({"goes"})) -> Case:
    return Case(0, answers, not answers, False, proposals)


GOOD = Scored("goes", -2.0, 0.9, 0)
LIKELY = Scored("gone", -1.0, 0.0, 0)
DELETE = Scored("", -0.5, 0.9, 1)


def test_normal_ignores_spacing() -> None:
    assert normal(" do n't ") == "don't"


def test_cases_read_each_proposal_and_leave_out_the_keep() -> None:
    readings = [reading(SENTENCE.tokens)]

    [found] = build_cases(
        [SENTENCE],
        gold_spans([SENTENCE]),
        readings,
        [""],
        Source(),
        Reader(),
        log=lambda _: None,
    )

    assert found.answers == {"goes"}
    assert found.keep_first
    assert [p.text for p in found.proposals] == ["go", "", "goes"]
    assert [round(p.drop, 1) for p in found.proposals] == [0.0, 0.9, 0.9]
    assert [p.removed for p in found.proposals] == [0, 1, 0]


def test_flagged_spans_follow_the_policy() -> None:
    spans = flagged_spans(TierPolicy())

    assert spans(0, reading(SENTENCE.tokens)) == [WordSpan(1, 2)]


def test_ranking_adds_the_tagger_and_the_size_penalty() -> None:
    both = case(LIKELY, GOOD, DELETE)

    assert ranked(both, 0.0, 0.0) == [DELETE, LIKELY, GOOD]
    assert ranked(both, 2.0, 0.0)[0] == DELETE
    assert ranked(both, 2.0, 2.0) == [GOOD, DELETE, LIKELY]


def test_recall_counts_unreachable_spans_and_skips_false_alarms() -> None:
    row = recall_row(
        "name",
        [
            case(LIKELY, GOOD),
            case(LIKELY, answers=frozenset({"unreachable"})),
            case(LIKELY, answers=frozenset()),
        ],
    )

    assert row.cases == 2
    assert row.recall[1] == 0.0
    assert row.recall[2] == 0.5


def test_tuning_finds_what_lifts_recall_and_prefers_no_change_on_a_tie() -> None:
    assert tune([case(DELETE, LIKELY, GOOD)]) == (0.0, 2.0)
    assert tune([case(GOOD, LIKELY)]) == (0.0, 0.0)


def test_contexts_come_from_earlier_sentences() -> None:
    assert contexts_of([SENTENCE, SENTENCE], 1) == ["", "He go home."]
