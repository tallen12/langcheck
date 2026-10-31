"""A small head on the frozen tagger: is what is missing at this gap a comma?

It trains only on gaps with a gold insertion, so it answers what is missing given that
something is, and the tagger's own scores and weights stay as they were.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.tagging.encoding import collate, encode, to_device
from langcheck.tagging.model import SlotTagger
from langcheck.tagging.training import InContext

COMMA = (",",)
BATCH_SIZE = 64
STEPS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
SEED = 0
_WEIGHTS = "comma.pt"


def has_comma_head(directory: Path) -> bool:
    """Whether ``directory`` holds a trained comma head."""
    return (directory / _WEIGHTS).is_file()


def save_comma_head(directory: Path, head: torch.nn.Linear) -> None:
    """Write ``head`` beside the tagger it was trained on."""
    torch.save(head.state_dict(), directory / _WEIGHTS)


def load_comma_head(
    directory: Path, tagger: SlotTagger, *, device: torch.device
) -> torch.nn.Linear:
    """The head saved in ``directory``, sized for ``tagger``, on ``device``."""
    head = torch.nn.Linear(tagger.gap_state_size, 1)
    head.load_state_dict(torch.load(directory / _WEIGHTS, map_location="cpu", weights_only=True))
    return head.to(device).eval()


def _batches(lengths: Sequence[int], size: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    return [order[begin : begin + size] for begin in range(0, len(order), size)]


def gap_labels(sentence: AnnotatedSentence) -> dict[int, bool]:
    """For each gap with a gold insertion, whether every insertion there is a comma."""
    labels: dict[int, bool] = {}
    for edits in sentence.annotators:
        for edit in edits:
            if edit.start == edit.end:
                comma = edit.correction == COMMA
                labels[edit.start] = labels.get(edit.start, True) and comma
    return labels


def train_comma_head(
    tagger: SlotTagger,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[InContext],
    *,
    device: torch.device,
    max_length: int,
    log: Callable[[str], None],
) -> torch.nn.Linear:
    """Fit the head on the gold insertions of ``items``. ``tagger`` does not change."""
    wanted = [(i.sentence.tokens, gap_labels(i.sentence), i.context) for i in items]
    wanted = [item for item in wanted if item[1]]
    if not wanted:
        msg = "no gold insertions to train on"
        raise ValueError(msg)
    encoded = [
        encode(tokens, tokenizer, max_length=max_length, context=context)
        for tokens, _, context in wanted
    ]
    pad = int(tokenizer.pad_token_id or 0)
    states: list[torch.Tensor] = []
    labels: list[float] = []
    tagger.eval()
    for indices in _batches([len(item.input_ids) for item in encoded], BATCH_SIZE):
        batch = to_device(collate([encoded[i] for i in indices], pad_token_id=pad), device)
        with torch.inference_mode():
            gaps = tagger.gap_states(batch).cpu()
        for row, index in enumerate(indices):
            for gap, comma in wanted[index][1].items():
                if gap < len(encoded[index].gap_left):
                    states.append(gaps[row, gap])
                    labels.append(float(comma))
    log(f"{len(labels)} gold insertions, {int(sum(labels))} of them commas")
    inputs = torch.stack(states).to(device)
    targets = torch.tensor(labels, device=device)
    torch.manual_seed(SEED)
    head = torch.nn.Linear(inputs.size(-1), 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for _ in range(STEPS):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head.forward(inputs).squeeze(-1), targets
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    log(f"final loss {loss.item():.4f}")
    return head.eval()


class CommaScorer:
    """Each gap's probability that what is missing there is a comma."""

    def __init__(
        self,
        tagger: SlotTagger,
        head: torch.nn.Linear,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        max_length: int,
    ) -> None:
        self._tagger = tagger
        self._head = head
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[tuple[float, ...]]:
        """One tuple of ``len(words) + 1`` per sentence; gaps past the limit score 0."""
        texts = contexts if contexts is not None else [""] * len(sentences)
        encoded = [
            encode(s, self._tokenizer, max_length=self._max_length, context=c)
            for s, c in zip(sentences, texts, strict=True)
        ]
        pad = int(self._tokenizer.pad_token_id or 0)
        results: list[tuple[float, ...]] = [()] * len(sentences)
        self._tagger.eval()
        for indices in _batches([len(item.input_ids) for item in encoded], BATCH_SIZE):
            batch = to_device(
                collate([encoded[i] for i in indices], pad_token_id=pad), self._device
            )
            with torch.inference_mode():
                logits = self._head.forward(self._tagger.gap_states(batch)).squeeze(-1)
            values = torch.sigmoid(logits).cpu().tolist()
            for row, index in enumerate(indices):
                kept = values[row][: len(encoded[index].gap_left)]
                missing = len(sentences[index]) + 1 - len(kept)
                results[index] = (*kept, *([0.0] * missing))
        return results
