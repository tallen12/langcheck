"""Judging corrections with the tagger: read each corrected sentence again, drop the poor
ones and order the rest by how much they lower the score.

A sentence's score is the sum of its flags' scores, not its highest slot: with two errors
in a sentence, fixing the lesser one would not move the highest slot at all. Flags, not
slots, because one error often lights up several neighbouring slots.
"""

from collections.abc import Iterable, Sequence

from langcheck.correction.models import (
    Candidate,
    CandidateOrder,
    Corrected,
    CorrectionPolicy,
    Pending,
    Ranked,
    Rejected,
    RejectionReason,
)
from langcheck.correction.place import corrected_sentence, touches
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import flag_sentences, in_text
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import TextFlag, Tier, TierPolicy


def sentence_score(flags: Iterable[TextFlag]) -> float:
    """How wrong the tagger finds a sentence with these flags."""
    return sum(flag.score for flag in flags)


def _others_after_edit(
    flags: Iterable[TextFlag], target: TextFlag, candidate: Candidate
) -> list[CharSpan]:
    """Where the sentence's other flags are once ``candidate`` is applied. A flag the edit
    swallows is left out: it is no longer there to explain a flag found afterwards.
    """
    change = len(candidate.replacement) - (candidate.span.end - candidate.span.start)
    others: list[CharSpan] = []
    for flag in flags:
        if flag == target:
            continue
        if flag.span.end <= candidate.span.start:
            others.append(flag.span)
        elif flag.span.start >= candidate.span.end:
            others.append(CharSpan(flag.span.start + change, flag.span.end + change))
    return others


def _judge(
    pending: Pending, candidate: Candidate, after: Sequence[TextFlag], policy: CorrectionPolicy
) -> Ranked | Rejected:
    """``after`` holds the corrected sentence's flags with offsets into the corrected text.
    Only error-tier flags count against a correction: the lower tier is mostly false alarms.
    """
    before = pending.flagged.flags
    score_before, score_after = sentence_score(before), sentence_score(after)
    reduction = score_before - score_after
    edit = CharSpan(candidate.span.start, candidate.span.start + len(candidate.replacement))
    others = _others_after_edit(before, pending.flag, candidate)
    unexplained = [
        flag
        for flag in after
        if flag.tier is Tier.ERROR and not any(touches(flag.span, other) for other in others)
    ]
    reason: RejectionReason | None = None
    if any(not touches(flag.span, edit) for flag in unexplained):
        reason = RejectionReason.NEW_ERROR
    elif unexplained:
        reason = RejectionReason.STILL_FLAGGED
    elif reduction < policy.min_reduction:
        reason = RejectionReason.NO_IMPROVEMENT
    if reason is not None:
        return Rejected(candidate.original, candidate.replacement, reason, reduction, candidate)
    return Ranked(candidate, score_before, score_after, policy.confidence(reduction))


def rank_corrections(
    pending: Sequence[Pending],
    reader: SentenceReader,
    *,
    tiers: TierPolicy,
    policy: CorrectionPolicy,
) -> list[Corrected]:
    """Each flag's corrections judged by the tagger, in one batch: at most ``policy.count``
    offered, ties keeping the corrector's order. Blocking.
    """
    placed = [(item, candidate) for item in pending for candidate in item.candidates]
    flagged_after = flag_sentences(
        [corrected_sentence(item.flagged.sentence, candidate) for item, candidate in placed],
        [item.flagged.sentence.context for item, _ in placed],
        reader,
        policy=tiers,
    )
    verdicts = iter(
        _judge(
            item,
            candidate,
            [in_text(flag, item.flagged.sentence.span.start) for flag in flags],
            policy,
        )
        for (item, candidate), flags in zip(placed, flagged_after, strict=True)
    )
    results: list[Corrected] = []
    for item in pending:
        ranked: list[Ranked] = []
        rejected = list(item.rejected)
        for _ in item.candidates:
            match next(verdicts):
                case Ranked() as kept:
                    ranked.append(kept)
                case Rejected() as dropped:
                    rejected.append(dropped)
        if policy.order is CandidateOrder.REDUCTION:
            ranked.sort(key=lambda r: r.reduction, reverse=True)
        results.append(Corrected(item.flag, tuple(ranked[: policy.count]), tuple(rejected)))
    return results
