"""``SentenceScorer`` by a classifier fine-tuned on CoLA (grammatical acceptability).

One pass per sentence gives ``log P(acceptable)`` for the whole sentence, which catches
agreement and word-order errors that PLL scores weakly. ``focus`` is ignored.
"""

from collections.abc import Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.modeling_utils import PreTrainedModel

from langcheck.scoring.models import ScoredText

DEFAULT_COLA_MODEL = "textattack/roberta-base-CoLA"
ACCEPTABLE_LABEL = 1
"""The class index of "acceptable" in CoLA-style models (0 is unacceptable)."""
BATCH_SIZE = 32
MAX_LENGTH = 128


class AcceptabilityScorer:
    """Scores each text by the log probability that it is acceptable English."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str,
        acceptable_label: int = ACCEPTABLE_LABEL,
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if batch_size < 1:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)
        self._device = torch.device(device)
        self._model = model.eval()
        # PreTrainedModel.to is wrapped in a way ty cannot follow; Module.to is the same call.
        torch.nn.Module.to(self._model, self._device)
        self._tokenizer = tokenizer
        self._label = acceptable_label
        self._batch_size = batch_size
        self._max_length = max_length

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """``log P(acceptable)`` of each text, in order."""
        scores: list[float] = []
        for start in range(0, len(texts), self._batch_size):
            scores += self._run([t.text for t in texts[start : start + self._batch_size]])
        return scores

    def _run(self, texts: list[str]) -> list[float]:
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self._model(
                input_ids=encoded["input_ids"].to(self._device),
                attention_mask=encoded["attention_mask"].to(self._device),
            ).logits
            acceptable = logits.float().log_softmax(dim=-1)[:, self._label]
        return [float(value) for value in acceptable.tolist()]


def load_acceptability_scorer(
    name: str = DEFAULT_COLA_MODEL, *, device: str
) -> AcceptabilityScorer:
    """Load a CoLA classifier and its tokenizer by Hub name (downloaded once, then cached)."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer is None:
        msg = f"no tokenizer found for {name}"
        raise ValueError(msg)
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype)
    return AcceptabilityScorer(model, tokenizer, device=device)
