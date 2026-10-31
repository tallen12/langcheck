"""``SlotScorer`` by a trained :class:`SlotTagger`."""

from collections.abc import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from langcheck.tagging.encoding import collate, encode, to_device
from langcheck.tagging.model import SlotTagger
from langcheck.tagging.models import SlotScores

BATCH_SIZE = 64


class TaggerScorer:
    """Scores sentences in batches of similar length, so little time goes on padding."""

    def __init__(
        self,
        tagger: SlotTagger,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        max_length: int,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._tagger = tagger
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        """One result per sentence, in order, each reading its context if given."""
        texts = contexts if contexts is not None else [""] * len(sentences)
        encoded = [
            encode(s, self._tokenizer, max_length=self._max_length, context=c)
            for s, c in zip(sentences, texts, strict=True)
        ]
        order = sorted(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
        results: list[SlotScores | None] = [None] * len(encoded)
        self._tagger.eval()
        for begin in range(0, len(order), self._batch_size):
            indices = order[begin : begin + self._batch_size]
            batch = collate(
                [encoded[index] for index in indices],
                pad_token_id=int(self._tokenizer.pad_token_id or 0),
            )
            with torch.inference_mode():
                word_logits, gap_logits = self._tagger(to_device(batch, self._device))
            words = torch.sigmoid(word_logits).cpu().tolist()
            gaps = torch.sigmoid(gap_logits).cpu().tolist()
            for row, index in enumerate(indices):
                results[index] = _scores(
                    words[row][: len(encoded[index].word_first)],
                    gaps[row][: len(encoded[index].gap_left)],
                    len(sentences[index]),
                )
        return [result for result in results if result is not None]


def _scores(words: list[float], gaps: list[float], length: int) -> SlotScores:
    """Slots the length limit left out score 0."""
    return SlotScores(
        (*words, *([0.0] * (length - len(words)))),
        (*gaps, *([0.0] * (length + 1 - len(gaps)))),
    )
