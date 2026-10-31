import asyncio
from collections.abc import Sequence

from langcheck.correction.models import (
    CandidateOrder,
    Corrected,
    CorrectionPolicy,
    FlaggedSentence,
    Pending,
)
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.proposal.corrector import ProposerCorrector, word_span
from langcheck.proposal.protocols import Proposal, ProposalRequest
from langcheck.proposal.targets import WordSpan
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.models import SentenceReading, SlotScores
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier, TierPolicy

POLICY = CorrectionPolicy(order=CandidateOrder.AS_WRITTEN)


class Reader:
    """Scores ``go`` as wrong, and nothing else."""

    def __init__(self) -> None:
        self.contexts: list[str] = []

    def read(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SentenceReading]:
        self.contexts += contexts or []
        return [
            SentenceReading(
                SlotScores(
                    tuple(0.9 if word == "go" else 0.0 for word in s), (0.0,) * (len(s) + 1)
                ),
                None,
            )
            for s in sentences
        ]


class Source:
    def __init__(self, *proposals: Proposal) -> None:
        self.proposals = proposals
        self.requests: list[ProposalRequest] = []

    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        self.requests += requests
        return [[self.proposals for _ in request.spans] for request in requests]


def flagged(start: int, end: int) -> FlaggedSentence:
    flag = TextFlag(CharSpan(start, end), 0.9, Tier.ERROR, FlagKind.GENERAL)
    return FlaggedSentence(SentenceInText(CharSpan(10, 20), "He go up", "Before."), (flag,))


def test_flags_map_to_words_and_gaps() -> None:
    spans = [CharSpan(0, 2), CharSpan(3, 5), CharSpan(6, 8)]

    assert word_span(spans, CharSpan(3, 5)) == WordSpan(1, 2)
    assert word_span(spans, CharSpan(0, 5)) == WordSpan(0, 2)
    assert word_span(spans, CharSpan(5, 5)) == WordSpan(2, 2)
    assert word_span(spans, CharSpan(2, 3)) == WordSpan(1, 1)


def test_requests_carry_the_sentence_span_and_context() -> None:
    source = Source(Proposal("goes", -1.0))

    asyncio.run(ProposerCorrector(source, Reader(), POLICY).correct([flagged(13, 15)]))

    assert source.requests == [ProposalRequest(("He", "go", "up"), "Before.", (WordSpan(1, 2),))]


def test_the_tagger_reorders_and_a_keep_is_dropped() -> None:
    source = Source(
        Proposal("", -0.1, keep=True),
        Proposal("go", -0.5),
        Proposal("goes", -2.0),
    )
    reader = Reader()

    [[pending]] = asyncio.run(ProposerCorrector(source, reader, POLICY).correct([flagged(13, 15)]))

    assert isinstance(pending, Pending)
    assert [c.replacement for c in pending.candidates] == ["goes"]
    assert [r.reason.value for r in pending.rejected] == ["no_change"]
    assert set(reader.contexts) == {"Before."}


def test_no_sentences_read_nothing() -> None:
    reader = Reader()

    assert asyncio.run(ProposerCorrector(Source(), reader, POLICY).correct([])) == []
    assert reader.contexts == []


def test_the_pipeline_offers_at_most_count_in_the_written_order() -> None:
    source = Source(
        Proposal("goes", -1.0),
        Proposal("went", -2.0),
        Proposal("gone", -3.0),
    )
    reader = Reader()
    policy = CorrectionPolicy(count=2, order=CandidateOrder.AS_WRITTEN)
    pipeline = CorrectionPipeline(
        corrector=ProposerCorrector(source, reader, policy),
        reader=reader,
        tiers=TierPolicy(),
        policy=policy,
        context_sentences=0,
    )

    [outcome] = asyncio.run(pipeline.suggest("He go up"))

    assert isinstance(outcome, Corrected)
    assert [r.candidate.replacement for r in outcome.ranked] == ["goes", "went"]


def test_a_word_becoming_punctuation_leaves_no_space_before_it() -> None:
    source = Source(Proposal(",", -0.1))

    [[pending]] = asyncio.run(
        ProposerCorrector(source, Reader(), POLICY).correct([flagged(13, 15)])
    )

    assert isinstance(pending, Pending)
    assert pending.candidates[0].span == CharSpan(12, 15)
