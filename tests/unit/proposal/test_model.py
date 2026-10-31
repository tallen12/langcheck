import torch
from transformers import PreTrainedTokenizerFast

from langcheck.proposal.model import (
    AT_GAP,
    INSIDE,
    OUTSIDE,
    ProposerShape,
    TokenProposer,
    pad_rows,
    span_fits,
    span_segments,
)
from langcheck.proposal.targets import WordSpan
from langcheck.tagging.encoding import encode

SHAPE = ProposerShape(
    encoder_size=16, embedding_size=16, vocab_size=20, size=8, layers=1, heads=2, dropout=0.0
)


def test_logits_cover_the_vocabulary_for_each_written_subword() -> None:
    proposer = TokenProposer(SHAPE).eval()
    tokens = torch.tensor([[0, 5, 6], [0, 7, 1]])

    logits = proposer(
        torch.zeros(2, 4, 16),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.long),
        tokens,
        tokens == 1,
        torch.rand(20, 16),
    )

    assert logits.shape == (2, 3, 20)


def test_later_subwords_do_not_change_earlier_logits() -> None:
    torch.manual_seed(0)
    proposer = TokenProposer(SHAPE).eval()
    embeddings = torch.rand(20, 16)
    states = torch.rand(1, 4, 16)

    def first_logits(tokens: list[int]) -> torch.Tensor:
        rows = torch.tensor([tokens])
        logits = proposer(
            states,
            torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, dtype=torch.long),
            rows,
            torch.zeros_like(rows, dtype=torch.bool),
            embeddings,
        )
        assert isinstance(logits, torch.Tensor)
        return logits[0, 0]

    assert torch.allclose(first_logits([0, 5]), first_logits([0, 9]), atol=1e-5)


def test_segments_mark_the_span_or_either_side_of_a_gap(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    encoded = encode(("The", "dog", "."), tiny_tokenizer, max_length=32)

    marks = span_segments(encoded, WordSpan(1, 2))
    assert INSIDE in marks
    assert marks[encoded.word_first[1]] == INSIDE
    assert marks[encoded.word_first[0]] == OUTSIDE

    gap = span_segments(encoded, WordSpan(1, 1))
    assert gap[encoded.gap_left[1]] == AT_GAP
    assert gap[encoded.gap_right[1]] == AT_GAP


def test_a_span_past_the_length_limit_does_not_fit(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    encoded = encode(("The", "dog", ".") * 10, tiny_tokenizer, max_length=8)

    assert span_fits(encoded, WordSpan(0, 1))
    assert not span_fits(encoded, WordSpan(25, 26))


def test_pad_rows_fills_short_rows() -> None:
    rows = pad_rows([[1], [2, 3]], 9, torch.device("cpu"))

    assert rows.tolist() == [[1, 9], [2, 3]]


def test_next_logits_are_the_last_of_the_full_logits() -> None:
    torch.manual_seed(0)
    proposer = TokenProposer(SHAPE).eval()
    embeddings = torch.rand(20, 16)
    states = torch.rand(2, 4, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    segments = torch.zeros(2, 4, dtype=torch.long)
    tokens = torch.tensor([[0, 5], [0, 6]])

    full = proposer(states, mask, segments, tokens, tokens == -1, embeddings)
    last = proposer.next_logits(states, mask, segments, tokens, embeddings)

    assert torch.allclose(full[:, -1], last, atol=1e-5)
