"""Training the proposer on the frozen tagger, keeping the epoch with the lowest loss."""

import copy
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from langcheck.proposal.model import (
    MAX_TARGET,
    TokenProposer,
    pad_rows,
    span_fits,
    span_segments,
    word_embeddings,
)
from langcheck.proposal.search import Vocabulary, target_ids, vocabulary
from langcheck.proposal.targets import Keep, Rewrite, Target
from langcheck.tagging.encoding import Encoded, collate, encode, to_device
from langcheck.tagging.model import SlotTagger

LOG_EVERY = 200


@dataclass(frozen=True, slots=True)
class ProposerOptions:
    """How to train."""

    epochs: int = 3
    batch_size: int = 16
    max_cases: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_share: float = 0.05
    max_grad_norm: float = 1.0
    max_length: int = 256
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ProposerItem:
    """A sentence, what it is read after, and what to write at its spans."""

    words: tuple[str, ...]
    context: str
    targets: tuple[Target, ...]


@dataclass(frozen=True, slots=True)
class _Prepared:
    encoded: Encoded
    cases: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    """The segments and target subwords of each usable target."""


@dataclass(frozen=True, slots=True)
class ProposerResult:
    """The best epoch and its validation loss."""

    epoch: int
    loss: float


def _prepare(
    item: ProposerItem,
    tokenizer: PreTrainedTokenizerBase,
    tokens: Vocabulary,
    *,
    max_length: int,
) -> _Prepared:
    encoded = encode(item.words, tokenizer, max_length=max_length, context=item.context)
    cases: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for target in item.targets:
        match target:
            case Rewrite(span=span, words=words):
                ids = target_ids(words, tokenizer, tokens)
            case Keep(span=span):
                ids = target_ids(None, tokenizer, tokens)
        if span_fits(encoded, span) and len(ids) <= MAX_TARGET:
            cases.append((span_segments(encoded, span), ids))
    return _Prepared(encoded, tuple(cases))


def batch_loss(
    tagger: SlotTagger,
    proposer: TokenProposer,
    prepared: Sequence[_Prepared],
    *,
    pad: int,
    device: torch.device,
    max_cases: int,
    shuffler: random.Random,
) -> torch.Tensor:
    """Cross-entropy over every written subword of up to ``max_cases`` targets."""
    batch = to_device(collate([p.encoded for p in prepared], pad_token_id=pad), device)
    with torch.no_grad():
        states = tagger.encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state
    cases = [
        (row, segments, ids) for row, item in enumerate(prepared) for segments, ids in item.cases
    ]
    if len(cases) > max_cases:
        cases = shuffler.sample(cases, max_cases)
    rows = torch.tensor([row for row, _, _ in cases], dtype=torch.long, device=device)
    segments = pad_rows([s for _, s, _ in cases], 0, device)
    written = pad_rows([ids for _, _, ids in cases], pad, device)
    width = batch.input_ids.size(1)
    segments = torch.nn.functional.pad(segments, (0, width - segments.size(1)))
    logits = proposer(
        states[rows],
        batch.attention_mask[rows].bool(),
        segments,
        written[:, :-1],
        written[:, :-1] == pad,
        word_embeddings(tagger),
    )
    answers = written[:, 1:]
    real = answers != pad
    return torch.nn.functional.cross_entropy(logits[real], answers[real])


def train_proposer(
    tagger: SlotTagger,
    proposer: TokenProposer,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[ProposerItem],
    validation: Sequence[ProposerItem],
    *,
    device: torch.device,
    options: ProposerOptions,
    log: Callable[[str], None],
) -> ProposerResult:
    """Fit ``proposer``, which ends with its best epoch's weights; ``tagger`` never changes."""
    torch.manual_seed(options.seed)
    shuffler = random.Random(options.seed)
    pad = int(tokenizer.pad_token_id or 0)
    tokens = vocabulary(tokenizer)
    length = options.max_length
    fit = [_prepare(item, tokenizer, tokens, max_length=length) for item in items]
    fit = [item for item in fit if item.cases]
    held = [_prepare(item, tokenizer, tokens, max_length=length) for item in validation]
    held = [item for item in held if item.cases]
    if not fit or not held:
        msg = "no targets to train or validate on"
        raise ValueError(msg)
    tagger.eval()
    for parameter in tagger.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        proposer.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    per_epoch = max(1, -(-len(fit) // options.batch_size))
    total = per_epoch * options.epochs
    warmup = max(1, int(total * options.warmup_share))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (total - step) / max(1, total - warmup))),
    )
    best: ProposerResult | None = None
    best_state: Mapping[str, object] = {}
    for epoch in range(1, options.epochs + 1):
        proposer.train()
        shuffler.shuffle(fit)
        for step, begin in enumerate(range(0, len(fit), options.batch_size), start=1):
            loss = batch_loss(
                tagger,
                proposer,
                fit[begin : begin + options.batch_size],
                pad=pad,
                device=device,
                max_cases=options.max_cases,
                shuffler=shuffler,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), options.max_grad_norm)
            optimizer.step()
            schedule.step()
            if step % LOG_EVERY == 0:
                log(f"epoch {epoch} step {step}/{per_epoch} loss {loss.item():.4f}")
        proposer.eval()
        losses: list[float] = []
        with torch.no_grad():
            for begin in range(0, len(held), options.batch_size):
                losses.append(
                    batch_loss(
                        tagger,
                        proposer,
                        held[begin : begin + options.batch_size],
                        pad=pad,
                        device=device,
                        max_cases=options.max_cases,
                        shuffler=shuffler,
                    ).item()
                )
        mean = sum(losses) / len(losses)
        log(f"epoch {epoch}: validation loss {mean:.4f}")
        if best is None or mean < best.loss:
            best = ProposerResult(epoch, mean)
            best_state = copy.deepcopy(proposer.state_dict())
    if best is None:
        msg = f"epochs must be positive, got {options.epochs}"
        raise ValueError(msg)
    proposer.load_state_dict(best_state)
    return best
