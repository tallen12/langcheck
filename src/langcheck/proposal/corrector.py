"""Corrections without an LLM: write proposals, have the tagger read them, keep the best."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from langcheck.correction.kind_guess import guess_kind
from langcheck.correction.models import CorrectionPolicy, FlaggedSentence, Pending, Unavailable
from langcheck.correction.place import place_corrections
from langcheck.correction.schemas import ReplyCorrection
from langcheck.proposal.protocols import Proposal, ProposalRequest, ProposalSource
from langcheck.proposal.rerank import RerankPolicy, Scored, reranked
from langcheck.proposal.targets import WordSpan
from langcheck.tagging.alignment import CharSpan, split_words, word_spans
from langcheck.tagging.models import SentenceReading
from langcheck.tagging.protocols import SentenceReader

JUDGED: Final = 8
"""Proposals per flag that reach the filters, so rejected ones leave others to offer."""


def word_span(spans: Sequence[CharSpan], flag: CharSpan) -> WordSpan:
    """The words of a sentence that ``flag`` covers; an empty flag is the gap at its place."""
    if flag.start == flag.end:
        gap = sum(span.end <= flag.start for span in spans)
        return WordSpan(gap, gap)
    inside = [i for i, span in enumerate(spans) if span.start < flag.end and flag.start < span.end]
    if not inside:
        gap = sum(span.end <= flag.start for span in spans)
        return WordSpan(gap, gap)
    return WordSpan(inside[0], inside[-1] + 1)


def _raw(reading: SentenceReading) -> float:
    return sum(reading.scores.words) + sum(reading.scores.gaps)


@dataclass(frozen=True, slots=True)
class ProposerCorrector:
    """A ``Corrector`` needing no server, so a flag is never unavailable. A "keep" answer is
    dropped rather than obeyed: it ranks first on too many real errors to hide a fix.
    """

    source: ProposalSource
    reader: SentenceReader
    policy: CorrectionPolicy
    rerank: RerankPolicy = field(default_factory=RerankPolicy)

    async def correct(
        self, flagged: Sequence[FlaggedSentence]
    ) -> list[Sequence[Pending | Unavailable]]:
        """For each sentence, an entry per flag, in order. The models run in a worker thread."""
        return await asyncio.to_thread(self._correct, flagged)

    def _correct(self, flagged: Sequence[FlaggedSentence]) -> list[Sequence[Pending | Unavailable]]:
        if not flagged:
            return []
        words = [split_words(item.sentence.text) for item in flagged]
        spans = [
            tuple(
                word_span(
                    word_spans(item.sentence.text, sentence),
                    CharSpan(
                        flag.span.start - item.sentence.span.start,
                        flag.span.end - item.sentence.span.start,
                    ),
                )
                for flag in item.flags
            )
            for item, sentence in zip(flagged, words, strict=True)
        ]
        written = self.source.propose(
            [
                ProposalRequest(sentence, item.sentence.context, wanted)
                for item, sentence, wanted in zip(flagged, words, spans, strict=True)
            ]
        )
        scored = self._score(flagged, words, spans, written)
        results: list[Sequence[Pending | Unavailable]] = []
        for item, per_flag in zip(flagged, scored, strict=True):
            entries: list[Pending | Unavailable] = []
            for flag, proposals in zip(item.flags, per_flag, strict=True):
                original = _flag_text(item, flag.span)
                corrections = [
                    _correction(original, s.text) for s in reranked(proposals, self.rerank)[:JUDGED]
                ]
                entries.append(
                    place_corrections(corrections, item, flag, policy=self.policy, limit=JUDGED)
                )
            results.append(entries)
        return results

    def _score(
        self,
        flagged: Sequence[FlaggedSentence],
        words: Sequence[tuple[str, ...]],
        spans: Sequence[tuple[WordSpan, ...]],
        written: Sequence[Sequence[tuple[Proposal, ...]]],
    ) -> list[list[list[Scored]]]:
        """Every proposal's score; each distinct corrected sentence is read once, with its own
        sentence's context, and originals along with them.
        """
        keys: dict[tuple[tuple[str, ...], str], int] = {}

        def key(sentence: tuple[str, ...], context: str) -> int:
            return keys.setdefault((sentence, context), len(keys))

        for item, sentence in zip(flagged, words, strict=True):
            key(sentence, item.sentence.context)
        planned: list[list[list[tuple[str, float, int, int]]]] = []
        for item, sentence, wanted, per_span in zip(flagged, words, spans, written, strict=True):
            flags: list[list[tuple[str, float, int, int]]] = []
            for span, proposals in zip(wanted, per_span, strict=True):
                row: list[tuple[str, float, int, int]] = []
                for proposal in proposals:
                    new = split_words(proposal.text)
                    after = (*sentence[: span.start], *new, *sentence[span.end :])
                    if proposal.keep or not after:
                        continue
                    removed = max(0, span.end - span.start - len(new))
                    row.append(
                        (
                            proposal.text,
                            proposal.log_probability,
                            key(after, item.sentence.context),
                            removed,
                        )
                    )
                flags.append(row)
            planned.append(flags)
        readings = self.reader.read([s for s, _ in keys], contexts=[c for _, c in keys])
        results: list[list[list[Scored]]] = []
        for item, sentence, flags in zip(flagged, words, planned, strict=True):
            before = _raw(readings[keys[(sentence, item.sentence.context)]])
            results.append(
                [
                    [
                        Scored(text, log_probability, before - _raw(readings[at]), removed)
                        for text, log_probability, at, removed in row
                    ]
                    for row in flags
                ]
            )
        return results


def _flag_text(item: FlaggedSentence, span: CharSpan) -> str:
    offset = item.sentence.span.start
    return item.sentence.text[span.start - offset : span.end - offset]


def _correction(original: str, text: str) -> ReplyCorrection:
    kind = guess_kind(original, text)
    return ReplyCorrection(
        original="",
        replacement=text,
        kind=kind.value,
        explanation=f"Proposed by the local model; looks like {kind.value}.",
    )
