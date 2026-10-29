"""``SentenceScorer`` by masked-LM pseudo-log-likelihood (Salazar et al., 2020).

Each token in a window around the edit is masked in turn and the model's log probability
of the true token is read. The score is the mean over the window, so a candidate is
compared with its original over the same stretch of text and a longer replacement is not
penalised for having more tokens. PLL prefers frequent words and shorter text, so only
the delta against the original means anything.

Model loading and batching follow ``contcheck.lm.hf_masked_lm``; RoBERTa-style models
(``distilroberta-base`` by default, DeBERTa-v3 as an option) work.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from langcheck.scoring.models import ScoredText

DEFAULT_PLL_MODEL = "distilroberta-base"
DEFAULT_WINDOW = 3
BATCH_SIZE = 64
MAX_LENGTH = 128


@dataclass(frozen=True, slots=True)
class _Masking:
    """One forward pass: ``input_ids`` with ``position`` masked, for text ``text_index``."""

    text_index: int
    input_ids: tuple[int, ...]
    position: int
    true_id: int


class PseudoLogLikelihoodScorer:
    """Scores texts by the mean log probability of the tokens around their focus."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str,
        window: int = DEFAULT_WINDOW,
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if window < 0 or batch_size < 1 or max_length < 3:
            msg = "window must be non-negative, batch_size positive and max_length at least 3"
            raise ValueError(msg)
        self._device = torch.device(device)
        self._model = model.eval()
        # PreTrainedModel.to is wrapped in a way ty cannot follow; Module.to is the same call.
        torch.nn.Module.to(self._model, self._device)
        self._tokenizer = tokenizer
        self._window = window
        self._batch_size = batch_size
        self._max_length = max_length
        self._mask_id = _token_id(tokenizer.mask_token_id, "MASK")
        self._pad_id = _token_id(tokenizer.pad_token_id, "PAD")

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """The mean log probability of each text's window tokens; 0.0 for empty text."""
        maskings = [m for index, text in enumerate(texts) for m in self._maskings(index, text)]
        totals = [0.0] * len(texts)
        counts = [0] * len(texts)
        for start in range(0, len(maskings), self._batch_size):
            batch = maskings[start : start + self._batch_size]
            for masking, log_probability in zip(batch, self._run(batch), strict=True):
                totals[masking.text_index] += log_probability
                counts[masking.text_index] += 1
        return [
            total / count if count else 0.0 for total, count in zip(totals, counts, strict=True)
        ]

    def _maskings(self, text_index: int, text: ScoredText) -> list[_Masking]:
        input_ids, offsets = self._encode(text.text)
        positions = window_positions(
            offsets, focus_start=text.focus.start, focus_end=text.focus.end, window=self._window
        )
        input_ids, positions = crop(input_ids, positions, max_length=self._max_length)
        return [
            _Masking(
                text_index,
                tuple(self._mask_id if i == position else t for i, t in enumerate(input_ids)),
                position,
                input_ids[position],
            )
            for position in positions
        ]

    def _encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._tokenizer(text, return_offsets_mapping=True)
        ids: object = encoded["input_ids"]
        offsets: object = encoded["offset_mapping"]
        if not isinstance(ids, list) or not isinstance(offsets, list):
            msg = "tokenizer returned no ids or offsets; a fast tokenizer is needed"
            raise TypeError(msg)
        return [int(i) for i in ids], [(int(s), int(e)) for s, e in offsets]

    def _run(self, batch: Sequence[_Masking]) -> list[float]:
        width = max(len(m.input_ids) for m in batch)
        input_ids = torch.full((len(batch), width), self._pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, masking in enumerate(batch):
            input_ids[row, : len(masking.input_ids)] = torch.tensor(masking.input_ids)
            attention[row, : len(masking.input_ids)] = 1
        rows = torch.arange(len(batch), device=self._device)
        positions = torch.tensor([m.position for m in batch], device=self._device)
        true_ids = torch.tensor([m.true_id for m in batch], device=self._device)
        with torch.inference_mode():
            logits = self._model(
                input_ids=input_ids.to(self._device), attention_mask=attention.to(self._device)
            ).logits
            log_probabilities = logits[rows, positions].float().log_softmax(dim=-1)
            picked = log_probabilities[rows, true_ids]
        return [float(value) for value in picked.tolist()]


def window_positions(
    offsets: Sequence[tuple[int, int]], *, focus_start: int, focus_end: int, window: int
) -> list[int]:
    """Indices of the tokens to mask: those overlapping the focus, plus ``window`` content
    tokens on each side. Special tokens (empty offsets) are never masked. An empty focus
    (a deletion) is anchored on the first token after it, or the last token.
    """
    content = [i for i, (start, end) in enumerate(offsets) if end > start]
    if not content:
        return []
    inside = [
        n
        for n, i in enumerate(content)
        if offsets[i][0] < focus_end and focus_start < offsets[i][1]
    ]
    if inside:
        low, high = inside[0], inside[-1]
    else:
        after = [n for n, i in enumerate(content) if offsets[i][0] >= focus_start]
        low = high = after[0] if after else len(content) - 1
    return content[max(low - window, 0) : high + window + 1]


def crop(
    input_ids: Sequence[int], positions: Sequence[int], *, max_length: int
) -> tuple[list[int], list[int]]:
    """``input_ids`` cut to ``max_length`` around the window, keeping the first and last
    (special) tokens, with ``positions`` moved to match. Plain truncation would cut off
    the end of a long sentence even when the edit is there.
    """
    if len(input_ids) <= max_length:
        return list(input_ids), list(positions)
    keep = max_length - 2
    center = positions[len(positions) // 2] if positions else 1
    start = min(max(center - keep // 2, 1), len(input_ids) - 1 - keep)
    cropped = [input_ids[0], *input_ids[start : start + keep], input_ids[-1]]
    moved = [p - start + 1 for p in positions if start <= p < start + keep]
    return cropped, moved


def _token_id(value: object, name: str) -> int:
    if not isinstance(value, int):
        msg = f"tokenizer has no {name} token"
        raise ValueError(msg)
    return value


def load_pll_scorer(
    name: str = DEFAULT_PLL_MODEL, *, device: str, window: int = DEFAULT_WINDOW
) -> PseudoLogLikelihoodScorer:
    """Load a masked LM and its tokenizer by Hub name (downloaded once, then cached).

    On a GPU the weights are half precision, which about halves the time per batch;
    log probabilities are still computed in float32.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer is None:
        msg = f"no tokenizer found for {name}"
        raise ValueError(msg)
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
    model = AutoModelForMaskedLM.from_pretrained(name, dtype=dtype)
    return PseudoLogLikelihoodScorer(model, tokenizer, device=device, window=window)
