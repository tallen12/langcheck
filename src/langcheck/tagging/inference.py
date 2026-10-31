"""``SlotScorer`` by a trained :class:`SlotTagger`."""

from collections.abc import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from langcheck.tagging.encoding import collate, encode, to_device
from langcheck.tagging.model import SlotTagger
from langcheck.tagging.models import SentenceReading, SlotScores

BATCH_SIZE = 64


class TaggerReader:
    """Reads sentences in batches of similar length, so little time goes on padding, with
    one encoder pass per batch however many heads read it.
    """

    def __init__(
        self,
        tagger: SlotTagger,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        max_length: int,
        comma_head: torch.nn.Linear | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._tagger = tagger
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._comma_head = comma_head
        self._batch_size = batch_size

    def read(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SentenceReading]:
        """One reading per sentence, in order, each reading its context if given."""
        texts = contexts if contexts is not None else [""] * len(sentences)
        encoded = [
            encode(s, self._tokenizer, max_length=self._max_length, context=c)
            for s, c in zip(sentences, texts, strict=True)
        ]
        order = sorted(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
        results: list[SentenceReading | None] = [None] * len(encoded)
        self._tagger.eval()
        for begin in range(0, len(order), self._batch_size):
            indices = order[begin : begin + self._batch_size]
            batch = collate(
                [encoded[index] for index in indices],
                pad_token_id=int(self._tokenizer.pad_token_id or 0),
            )
            with torch.inference_mode():
                word_logits, gap_logits, gap_states = self._tagger.read(
                    to_device(batch, self._device)
                )
                commas = (
                    torch.sigmoid(self._comma_head.forward(gap_states).squeeze(-1)).cpu().tolist()
                    if self._comma_head is not None
                    else None
                )
            words = torch.sigmoid(word_logits).cpu().tolist()
            gaps = torch.sigmoid(gap_logits).cpu().tolist()
            for row, index in enumerate(indices):
                length = len(sentences[index])
                kept = len(encoded[index].gap_left)
                results[index] = SentenceReading(
                    _scores(words[row][: len(encoded[index].word_first)], gaps[row][:kept], length),
                    (
                        (*commas[row][:kept], *([0.0] * (length + 1 - kept)))
                        if commas is not None
                        else None
                    ),
                )
        return [result for result in results if result is not None]


class TaggerScorer:
    """The slot scores of a :class:`TaggerReader`, for evaluating them alone."""

    def __init__(
        self,
        tagger: SlotTagger,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        max_length: int,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._reader = TaggerReader(
            tagger, tokenizer, device=device, max_length=max_length, batch_size=batch_size
        )

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        """One result per sentence, in order, each reading its context if given."""
        return [reading.scores for reading in self._reader.read(sentences, contexts=contexts)]


def _scores(words: list[float], gaps: list[float], length: int) -> SlotScores:
    """Slots the length limit left out score 0."""
    return SlotScores(
        (*words, *([0.0] * (length - len(words)))),
        (*gaps, *([0.0] * (length + 1 - len(gaps)))),
    )
