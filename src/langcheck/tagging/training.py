"""Fine-tuning a :class:`SlotTagger`, keeping the epoch that finds the most errors, slot
by slot or by the flags a reader would see.
"""

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import assert_never

import torch
from transformers import PreTrainedTokenizerBase

from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.evaluation.models import ScoredEdit
from langcheck.tagging.encoding import collate, encode, to_device
from langcheck.tagging.flag_reach import FlagReach, flag_reach
from langcheck.tagging.inference import TaggerScorer
from langcheck.tagging.labels import SlotLabels, slot_labels
from langcheck.tagging.metrics import Measurement, ScoredSentence, operating_point
from langcheck.tagging.model import SlotTagger, slot_loss
from langcheck.tagging.models import SlotScores
from langcheck.tagging.selection import ByFlags, BySlots, EpochChoice
from langcheck.tagging.tiers import TierPolicy

HOLDOUT_BLOCK = 100
HOLDOUT_EVERY = 20
"""Hold out one block of sentences in this many: neighbouring sentences come from one
essay, so a random split would leak its writer and topic into validation."""
LOG_EVERY = 200


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    """How to fine-tune. ``precision`` is the operating point epochs are judged at."""

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_share: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 256
    precision: float = 0.5
    seed: int = 0
    choice: EpochChoice = field(default_factory=BySlots)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The best epoch, its operating point on the validation sentences and, when epochs
    are judged by flags, how its flags did.
    """

    epoch: int
    validation: Measurement
    flags: FlagReach | None = None


@dataclass(frozen=True, slots=True)
class InContext:
    """A sentence and the text before it, read but never scored."""

    sentence: AnnotatedSentence
    context: str = ""


def split_holdout[T](sentences: Sequence[T]) -> tuple[list[T], list[T]]:
    """``sentences`` as training and validation, split in blocks."""
    train: list[T] = []
    held: list[T] = []
    for index, sentence in enumerate(sentences):
        block = index // HOLDOUT_BLOCK
        (held if block % HOLDOUT_EVERY == HOLDOUT_EVERY - 1 else train).append(sentence)
    return train, held


def sentence_labels(sentence: AnnotatedSentence) -> SlotLabels:
    """A slot is wrong when any annotator edited it."""
    spans = [(edit.start, edit.end) for edits in sentence.annotators for edit in edits]
    return slot_labels(len(sentence.tokens), spans)


def validate(scorer: TaggerScorer, items: Sequence[InContext], *, precision: float) -> Measurement:
    """The best operating point on ``sentences`` at ``precision``."""
    return operating_on(_scores(scorer, items), items, precision=precision)


def _scores(scorer: TaggerScorer, items: Sequence[InContext]) -> list[SlotScores]:
    return scorer.score(
        [item.sentence.tokens for item in items], contexts=[item.context for item in items]
    )


def operating_on(
    scores: Sequence[SlotScores], items: Sequence[InContext], *, precision: float
) -> Measurement:
    """The best operating point at ``precision`` of ``scores`` for ``items``."""
    sentences = [item.sentence for item in items]
    scored = [
        ScoredSentence(
            score,
            tuple(
                tuple(ScoredEdit(e.start, e.end, e.correction, None) for e in edits)
                for edits in sentence.annotators
            ),
        )
        for score, sentence in zip(scores, sentences, strict=True)
    ]
    return operating_point(scored, precision=precision)


def train(
    tagger: SlotTagger,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[InContext],
    validation: Sequence[InContext],
    *,
    device: torch.device,
    options: TrainingOptions,
    log: Callable[[str], None],
) -> TrainingResult:
    """Fine-tune ``tagger`` on ``items``; it ends with its best epoch's weights."""
    if not validation:
        msg = "no validation sentences: a threshold cannot be chosen without them"
        raise ValueError(msg)
    torch.manual_seed(options.seed)
    shuffler = random.Random(options.seed)
    pad = int(tokenizer.pad_token_id or 0)
    encoded = [
        encode(i.sentence.tokens, tokenizer, max_length=options.max_length, context=i.context)
        for i in items
    ]
    labels = [sentence_labels(i.sentence) for i in items]
    scorer = TaggerScorer(tagger, tokenizer, device=device, max_length=options.max_length)
    optimizer = torch.optim.AdamW(
        tagger.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    per_epoch = max(1, -(-len(encoded) // options.batch_size))
    total = per_epoch * options.epochs
    warmup = max(1, int(total * options.warmup_share))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (total - step) / max(1, total - warmup))),
    )
    best: TrainingResult | None = None
    best_rating = 0.0
    best_state: Mapping[str, object] = {}
    for epoch in range(1, options.epochs + 1):
        tagger.train()
        order = list(range(len(encoded)))
        shuffler.shuffle(order)
        for step, begin in enumerate(range(0, len(order), options.batch_size), start=1):
            indices = order[begin : begin + options.batch_size]
            batch = to_device(
                collate(
                    [encoded[i] for i in indices],
                    pad_token_id=pad,
                    labels=[labels[i] for i in indices],
                ),
                device,
            )
            word_logits, gap_logits = tagger(batch)
            loss = slot_loss(batch, word_logits, gap_logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), options.max_grad_norm)
            optimizer.step()
            schedule.step()
            if step % LOG_EVERY == 0:
                log(f"epoch {epoch} step {step}/{per_epoch} loss {loss.item():.4f}")
        scores = _scores(scorer, validation)
        point = operating_on(scores, validation, precision=options.precision)
        log(
            f"epoch {epoch}: threshold {point.threshold:.2f} "
            f"precision {point.slots.precision:.3f} recall {point.slots.recall:.3f}"
        )
        match options.choice:
            case BySlots():
                found = TrainingResult(epoch, point)
                rating = point.slots.recall
            case ByFlags(policy=policy):
                flags = flag_reach([item.sentence for item in validation], scores, policy)
                log(flag_line(epoch, policy, flags))
                found = TrainingResult(epoch, point, flags)
                rating = flags.f05
            case _:
                assert_never(options.choice)
        if best is None or rating > best_rating:
            best, best_rating = found, rating
            best_state = copy.deepcopy(tagger.state_dict())
    if best is None:
        msg = f"epochs must be positive, got {options.epochs}"
        raise ValueError(msg)
    tagger.load_state_dict(best_state)
    log(f"keeping epoch {best.epoch}")
    return best


def flag_line(epoch: int, policy: TierPolicy, flags: FlagReach) -> str:
    """One epoch's flags on the validation sentences."""
    return (
        f"epoch {epoch}: flags from {policy.suggestion:.2f}, {flags.flags} flags, precision "
        f"{flags.precision:.3f}, {flags.reached}/{flags.edits} edits inside a flag "
        f"(reach {flags.reach:.3f}), F0.5 {flags.f05:.4f}"
    )
