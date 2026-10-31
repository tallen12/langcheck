"""The tagger: a transformer encoder with one error head for words and one for gaps."""

from typing import override

import torch
from transformers.modeling_utils import PreTrainedModel

from langcheck.tagging.encoding import Batch

DROPOUT = 0.1


class SlotTagger(torch.nn.Module):
    """Gives each word and gap a logit, read from the two subwords that bound it."""

    def __init__(self, encoder: PreTrainedModel) -> None:
        super().__init__()
        self.encoder = encoder
        size = encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.word_head = torch.nn.Linear(2 * size, 1)
        self.gap_head = torch.nn.Linear(2 * size, 1)

    @override
    def forward(  # ty: ignore[invalid-mutable-override]  # torch types forward as an attribute
        self, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for words and for gaps, shaped like ``batch``'s word and gap masks."""
        hidden = self.encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state
        hidden = self.dropout(hidden)
        words = torch.cat([_pick(hidden, batch.word_first), _pick(hidden, batch.word_last)], dim=-1)
        gaps = torch.cat([_pick(hidden, batch.gap_left), _pick(hidden, batch.gap_right)], dim=-1)
        return self.word_head.forward(words).squeeze(-1), self.gap_head.forward(gaps).squeeze(-1)


def _pick(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The hidden states at ``positions`` (batch by slots) of each sentence."""
    return hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))


def slot_loss(batch: Batch, word_logits: torch.Tensor, gap_logits: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over all real slots, words and gaps alike."""
    logits = torch.cat([word_logits[batch.word_mask], gap_logits[batch.gap_mask]])
    labels = torch.cat([batch.word_labels[batch.word_mask], batch.gap_labels[batch.gap_mask]])
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
