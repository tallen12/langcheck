"""Turning sentences into model input, keeping which subwords belong to which word."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from langcheck.evaluation.detokenize import detokenize
from langcheck.tagging.alignment import word_spans
from langcheck.tagging.labels import SlotLabels


@dataclass(frozen=True, slots=True)
class Encoded:
    """A sentence's subword ids and the subwords around each slot it can score.

    Words are read from their first and last subword and gaps from the subwords either
    side of them, the sentence's start and end tokens at its edges. Words past the length
    limit are left out, so there may be fewer entries than the sentence has slots.
    """

    input_ids: tuple[int, ...]
    word_first: tuple[int, ...]
    word_last: tuple[int, ...]
    gap_left: tuple[int, ...]
    gap_right: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Batch:
    """Padded sentences. Masks say which entries are real; labels only when training."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    word_first: torch.Tensor
    word_last: torch.Tensor
    word_mask: torch.Tensor
    gap_left: torch.Tensor
    gap_right: torch.Tensor
    gap_mask: torch.Tensor
    word_labels: torch.Tensor
    gap_labels: torch.Tensor


def load_tokenizer(name: str | Path) -> PreTrainedTokenizerBase:
    """The fast tokenizer of ``name``; offsets into the text need a fast one."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase) or not tokenizer.is_fast:
        msg = f"{name} has no fast tokenizer"
        raise ValueError(msg)
    return tokenizer


def encode(
    words: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    context: str = "",
) -> Encoded:
    """``words`` encoded with normal spacing, as a writer's text would be: the sets'
    space-separated tokens would teach the model their spacing, not grammar.

    ``context`` goes before them to be read, never scored: it is cut from its start to fit
    the limit, keeping what is nearest, and every position shifts past it.
    """
    text = detokenize(words)
    spans = word_spans(text, words)
    encoding = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    input_ids = [int(token) for token in encoding["input_ids"]]
    # Special tokens have empty offsets, as do pieces of characters a tokenizer splits.
    offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
    subwords = [index for index, (start, end) in enumerate(offsets) if end > start]
    first: list[int] = []
    last: list[int] = []
    cursor = 0
    for span in spans:
        while cursor < len(subwords) and offsets[subwords[cursor]][1] <= span.start:
            cursor += 1
        if cursor == len(subwords):
            break
        # A word with no subword of its own (an unusual character) reads the next one.
        end = cursor
        while end + 1 < len(subwords) and offsets[subwords[end + 1]][0] < span.end:
            end += 1
        first.append(subwords[cursor])
        last.append(subwords[end])
    complete = len(first) == len(words)
    left = [0, *last] if complete else [0, *last[:-1]]
    right = [*first, len(input_ids) - 1] if complete else first
    plain = Encoded(tuple(input_ids), tuple(first), tuple(last), tuple(left), tuple(right))
    return _with_context(plain, context, tokenizer, max_length=max_length) if context else plain


def _with_context(
    encoded: Encoded, context: str, tokenizer: PreTrainedTokenizerBase, *, max_length: int
) -> Encoded:
    """``encoded`` as ``<s> context </s></s> sentence </s>``, the pair form RoBERTa expects."""
    separator = int(tokenizer.sep_token_id or 0)
    room = max_length - len(encoded.input_ids) - 2
    if room <= 0:
        return encoded
    ids = [int(token) for token in tokenizer(context, add_special_tokens=False)["input_ids"]]
    kept = ids[-room:]
    shift = len(kept) + 2
    return Encoded(
        (encoded.input_ids[0], *kept, separator, separator, *encoded.input_ids[1:]),
        tuple(position + shift for position in encoded.word_first),
        tuple(position + shift for position in encoded.word_last),
        tuple(position + shift for position in encoded.gap_left),
        tuple(position + shift for position in encoded.gap_right),
    )


def _pad(rows: Sequence[Sequence[int]], value: int = 0) -> torch.Tensor:
    width = max((len(row) for row in rows), default=0)
    return torch.tensor(
        [[*row, *([value] * (width - len(row)))] for row in rows], dtype=torch.long
    ).reshape(len(rows), width)


def collate(
    encoded: Sequence[Encoded],
    *,
    pad_token_id: int,
    labels: Sequence[SlotLabels] | None = None,
) -> Batch:
    """``encoded`` as tensors. Labels past a sentence's length limit are dropped."""
    word_labels = [
        label.words[: len(item.word_first)]
        for item, label in zip(encoded, labels or [], strict=False)
    ]
    gap_labels = [
        label.gaps[: len(item.gap_left)] for item, label in zip(encoded, labels or [], strict=False)
    ]
    return Batch(
        input_ids=_pad([item.input_ids for item in encoded], pad_token_id),
        attention_mask=_pad([[1] * len(item.input_ids) for item in encoded]),
        word_first=_pad([item.word_first for item in encoded]),
        word_last=_pad([item.word_last for item in encoded]),
        word_mask=_pad([[1] * len(item.word_first) for item in encoded]).bool(),
        gap_left=_pad([item.gap_left for item in encoded]),
        gap_right=_pad([item.gap_right for item in encoded]),
        gap_mask=_pad([[1] * len(item.gap_left) for item in encoded]).bool(),
        word_labels=_pad([[int(v) for v in row] for row in word_labels]).float(),
        gap_labels=_pad([[int(v) for v in row] for row in gap_labels]).float(),
    )


def to_device(batch: Batch, device: torch.device) -> Batch:
    """``batch`` with its tensors on ``device``."""
    return Batch(
        batch.input_ids.to(device),
        batch.attention_mask.to(device),
        batch.word_first.to(device),
        batch.word_last.to(device),
        batch.word_mask.to(device),
        batch.gap_left.to(device),
        batch.gap_right.to(device),
        batch.gap_mask.to(device),
        batch.word_labels.to(device),
        batch.gap_labels.to(device),
    )
