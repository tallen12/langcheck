"""The proposer: a small decoder writing subwords, reading the frozen tagger's states.

It owns no vocabulary: input and output share the encoder's word embeddings, passed in, so
its checkpoint stays small and one encoder pass serves tagging and proposing alike.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch

from langcheck.proposal.targets import WordSpan
from langcheck.tagging.encoding import Encoded
from langcheck.tagging.model import SlotTagger

OUTSIDE = 0
INSIDE = 1
AT_GAP = 2
SEGMENTS = 3
MAX_TARGET = 16
"""Subwords a proposal may hold, counting its start and end tokens."""


@dataclass(frozen=True, slots=True)
class ProposerShape:
    """The decoder's size. ``encoder_size`` and ``embedding_size`` come from the encoder."""

    encoder_size: int
    embedding_size: int
    vocab_size: int
    size: int = 512
    layers: int = 2
    heads: int = 8
    dropout: float = 0.1


class TokenProposer(torch.nn.Module):
    """Gives logits over the next subword of the text belonging at a marked span."""

    def __init__(self, shape: ProposerShape) -> None:
        super().__init__()
        self.memory_in = torch.nn.Linear(shape.encoder_size, shape.size)
        self.segments = torch.nn.Embedding(SEGMENTS, shape.size)
        self.token_in = torch.nn.Linear(shape.embedding_size, shape.size)
        self.positions = torch.nn.Embedding(MAX_TARGET, shape.size)
        layer = torch.nn.TransformerDecoderLayer(
            shape.size,
            shape.heads,
            dim_feedforward=4 * shape.size,
            dropout=shape.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(layer, shape.layers)
        self.token_out = torch.nn.Linear(shape.size, shape.embedding_size)
        self.bias = torch.nn.Parameter(torch.zeros(shape.vocab_size))

    @override
    def forward(  # ty: ignore[invalid-mutable-override]  # torch types forward as an attribute
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        segments: torch.Tensor,
        tokens: torch.Tensor,
        tokens_padding: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Logits (cases by subwords by vocabulary) for the subword after each of ``tokens``.
        ``states_mask`` is true where a state is real, ``tokens_padding`` where a token is not.
        """
        decoded = self._decode(states, states_mask, segments, tokens, tokens_padding, embeddings)
        return self.token_out.forward(decoded) @ embeddings.T + self.bias

    def next_logits(
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        segments: torch.Tensor,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Logits (cases by vocabulary) for the subword after unpadded ``tokens``. Searching
        needs only these, and the vocabulary is too wide to spend on the earlier subwords.
        """
        padding = torch.zeros_like(tokens, dtype=torch.bool)
        decoded = self._decode(states, states_mask, segments, tokens, padding, embeddings)
        return self.token_out.forward(decoded[:, -1, :]) @ embeddings.T + self.bias

    def _decode(
        self,
        states: torch.Tensor,
        states_mask: torch.Tensor,
        segments: torch.Tensor,
        tokens: torch.Tensor,
        tokens_padding: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.memory_in.forward(states) + self.segments.forward(segments)
        length = tokens.size(1)
        steps = torch.arange(length, device=tokens.device).unsqueeze(0)
        written = self.token_in.forward(embeddings[tokens]) + self.positions.forward(steps)
        causal = torch.nn.Transformer.generate_square_subsequent_mask(
            length, device=tokens.device, dtype=torch.bool
        )
        return self.decoder.forward(
            written,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tokens_padding,
            memory_key_padding_mask=~states_mask,
        )


def word_embeddings(tagger: SlotTagger) -> torch.Tensor:
    """The word embeddings of ``tagger``'s encoder, which proposers write with."""
    weight = tagger.encoder.get_input_embeddings().weight
    if not isinstance(weight, torch.Tensor):
        msg = "the encoder's input embeddings have no weight tensor"
        raise TypeError(msg)
    return weight.detach()


def span_fits(encoded: Encoded, span: WordSpan) -> bool:
    """Whether the length limit left ``span``'s words and edges in the encoded sentence."""
    return span.end < len(encoded.gap_left) and span.start < len(encoded.gap_right)


def span_segments(encoded: Encoded, span: WordSpan) -> tuple[int, ...]:
    """A marker per subword: inside the span, or either side of the empty span's gap."""
    marks = [OUTSIDE] * len(encoded.input_ids)
    if span.start == span.end:
        marks[encoded.gap_left[span.start]] = AT_GAP
        marks[encoded.gap_right[span.start]] = AT_GAP
    else:
        for position in range(encoded.word_first[span.start], encoded.word_last[span.end - 1] + 1):
            marks[position] = INSIDE
    return tuple(marks)


def pad_rows(rows: Sequence[Sequence[int]], value: int, device: torch.device) -> torch.Tensor:
    """``rows`` as one tensor, shorter rows filled with ``value``."""
    width = max(len(row) for row in rows)
    return torch.tensor(
        [[*row, *([value] * (width - len(row)))] for row in rows],
        dtype=torch.long,
        device=device,
    )
