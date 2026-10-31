"""Measuring proposals against references: does a right answer appear in the top k, first
in the proposer's own order and then re-ranked by what the tagger makes of each sentence.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.proposal.protocols import Proposal, ProposalRequest, ProposalSource
from langcheck.proposal.targets import WordSpan, flag_span, is_false_alarm, references
from langcheck.tagging.alignment import preceding_text, split_words
from langcheck.tagging.models import SentenceReading
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import TierPolicy, slot_flags

CUTOFFS = (1, 2, 4, 8, 16)
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PENALTIES = (0.0, 0.5, 1.0, 2.0, 4.0)
TUNING_CUTOFF = 2
"""The default number of corrections offered per flag."""
LOG_EVERY = 500
_SPACE = re.compile(r"\s+")


def normal(text: str) -> str:
    """``text`` without whitespace, so tokenization differences cannot decide a match."""
    return _SPACE.sub("", text)


@dataclass(frozen=True, slots=True)
class Scored:
    """A proposal with what the tagger thinks: how much it lowers the summed slot
    probabilities and how many words it removes from the span.
    """

    text: str
    log_probability: float
    drop: float
    removed: int


@dataclass(frozen=True, slots=True)
class Case:
    """Proposals at one span. ``answers`` is empty when no right answer can be written from
    here, and ``false_alarm`` says no annotator edited anything near it.
    """

    sentence: int
    answers: frozenset[str]
    false_alarm: bool
    keep_first: bool
    proposals: tuple[Scored, ...]


class RecallRow(BaseModel):
    """Share of spans with a right answer in the top k, under one scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scoring: str
    cases: int
    recall: dict[int, float]
    deletions_first: float


def _raw(reading: SentenceReading) -> float:
    return sum(reading.scores.words) + sum(reading.scores.gaps)


def build_cases(
    sentences: Sequence[AnnotatedSentence],
    spans_of: Callable[[int, SentenceReading], Sequence[WordSpan]],
    readings: Sequence[SentenceReading],
    contexts: Sequence[str],
    source: ProposalSource,
    reader: SentenceReader,
    *,
    log: Callable[[str], None],
) -> list[Case]:
    """A case for every span, with every distinct proposal read by the tagger."""
    pending: list[tuple[int, WordSpan, tuple[Proposal, ...]]] = []
    wanted = [(index, tuple(spans_of(index, readings[index]))) for index in range(len(sentences))]
    wanted = [item for item in wanted if item[1]]
    for begin in range(0, len(wanted), LOG_EVERY):
        part = wanted[begin : begin + LOG_EVERY]
        written = source.propose(
            [ProposalRequest(sentences[i].tokens, contexts[i], spans) for i, spans in part]
        )
        for (index, spans), per_span in zip(part, written, strict=True):
            pending += [(index, s, p) for s, p in zip(spans, per_span, strict=True)]
        log(f"proposed for {begin + len(part)}/{len(wanted)} sentences")
    keys: dict[tuple[str, ...], int] = {}
    owners: list[int] = []
    rows: list[list[tuple[str, float, int, int]]] = []
    for index, span, proposals in pending:
        tokens = sentences[index].tokens
        row: list[tuple[str, float, int, int]] = []
        for proposal in proposals:
            if proposal.keep:
                continue
            new = split_words(proposal.text)
            after = (*tokens[: span.start], *new, *tokens[span.end :])
            if not after:
                continue
            removed = max(0, span.end - span.start - len(new))
            if after not in keys:
                keys[after] = len(owners)
                owners.append(index)
            row.append((proposal.text, proposal.log_probability, keys[after], removed))
        rows.append(row)
    log(f"reading {len(keys)} corrected sentences")
    after_read = reader.read(list(keys), contexts=[contexts[owner] for owner in owners])
    cases: list[Case] = []
    for (index, span, proposals), row in zip(pending, rows, strict=True):
        sentence = sentences[index]
        before = _raw(readings[index])
        cases.append(
            Case(
                sentence=index,
                answers=frozenset(normal("".join(words)) for words in references(sentence, span)),
                false_alarm=is_false_alarm(sentence, span),
                keep_first=bool(proposals) and proposals[0].keep,
                proposals=tuple(
                    Scored(text, lp, before - _raw(after_read[key]), removed)
                    for text, lp, key, removed in row
                ),
            )
        )
    return cases


def flagged_spans(policy: TierPolicy) -> Callable[[int, SentenceReading], list[WordSpan]]:
    """The spans of a sentence's flags under ``policy``, as users would see them."""

    def spans(_: int, reading: SentenceReading) -> list[WordSpan]:
        return [flag_span(flag) for flag in slot_flags(reading.scores, reading.commas, policy)]

    return spans


def gold_spans(
    sentences: Sequence[AnnotatedSentence],
) -> Callable[[int, SentenceReading], list[WordSpan]]:
    """Each gold edit's own span: what proposals could do if flags were exact."""

    def spans(index: int, _: SentenceReading) -> list[WordSpan]:
        found = {
            WordSpan(edit.start, edit.end)
            for annotator in sentences[index].annotators
            for edit in annotator
            if edit.code != "UNK"
        }
        return sorted(found, key=lambda s: (s.start, s.end))

    return spans


def ranked(case: Case, weight: float, penalty: float) -> list[Scored]:
    """``case``'s proposals under ``log probability + weight * drop - penalty * removed``."""
    return sorted(
        case.proposals,
        key=lambda s: s.log_probability + weight * s.drop - penalty * s.removed,
        reverse=True,
    )


def recall_row(
    scoring: str, cases: Iterable[Case], *, weight: float = 0.0, penalty: float = 0.0
) -> RecallRow:
    """Recall at each cutoff over ``cases``; unreachable spans count as misses, and false
    alarms are left out.
    """
    wanted = [case for case in cases if not case.false_alarm]
    hits = dict.fromkeys(CUTOFFS, 0)
    deletions = 0
    for case in wanted:
        order = ranked(case, weight, penalty)
        deletions += bool(order) and order[0].removed > 0
        where = next(
            (at for at, s in enumerate(order, start=1) if normal(s.text) in case.answers), None
        )
        for cutoff in CUTOFFS:
            hits[cutoff] += where is not None and where <= cutoff
    size = max(1, len(wanted))
    return RecallRow(
        scoring=scoring,
        cases=len(wanted),
        recall={cutoff: hits[cutoff] / size for cutoff in CUTOFFS},
        deletions_first=deletions / size,
    )


def tune(cases: Sequence[Case]) -> tuple[float, float]:
    """The weight and penalty with the best recall at ``TUNING_CUTOFF``; ties go to the
    smallest, so the tagger gets no more say than it earns.
    """
    best = (0.0, 0.0)
    top = -1.0
    for weight in WEIGHTS:
        for penalty in PENALTIES:
            value = recall_row("", cases, weight=weight, penalty=penalty).recall[TUNING_CUTOFF]
            if value > top:
                best, top = (weight, penalty), value
    return best


def contexts_of(sentences: Sequence[AnnotatedSentence], count: int) -> list[str]:
    """What each of ``sentences`` is read after."""
    return preceding_text([s.tokens for s in sentences], count)
