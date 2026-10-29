import math

import pytest
import torch
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM

from langcheck.scoring.models import ScoredText
from langcheck.scoring.pll import PseudoLogLikelihoodScorer, crop, window_positions
from langcheck.scoring.protocols import SentenceScorer
from langcheck.text.models import Span


def make_scorer(
    tokenizer: PreTrainedTokenizerFast, *, batch_size: int = 64, window: int = 2
) -> PseudoLogLikelihoodScorer:
    """A random two-layer RoBERTa, seeded so scores repeat."""
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    return PseudoLogLikelihoodScorer(
        RobertaForMaskedLM(config),
        tokenizer,
        device="cpu",
        window=window,
        batch_size=batch_size,
        max_length=32,
    )


ORIGINAL = ScoredText("I think their going home.", Span(8, 19))
CANDIDATE = ScoredText("I think they're going home.", Span(8, 21))


def test_scores_are_mean_log_probabilities(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    scorer: SentenceScorer = make_scorer(tiny_tokenizer)

    original, candidate = scorer.score([ORIGINAL, CANDIDATE])

    assert all(math.isfinite(s) and s < 0 for s in (original, candidate))
    assert original != candidate


def test_batching_does_not_change_scores(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    together = make_scorer(tiny_tokenizer).score([ORIGINAL, CANDIDATE])
    one_by_one = make_scorer(tiny_tokenizer, batch_size=1).score([ORIGINAL, CANDIDATE])

    assert one_by_one == pytest.approx(together, abs=1e-5)


def test_empty_text_scores_zero(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    assert make_scorer(tiny_tokenizer).score([ScoredText("", Span(0, 0))]) == [0.0]
    assert make_scorer(tiny_tokenizer).score([]) == []


def test_rejects_bad_settings(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="window"):
        make_scorer(tiny_tokenizer, window=-1)


# Offsets of "<s> I think their going home . </s>" as a fast tokenizer reports them.
OFFSETS = [(0, 0), (0, 1), (2, 7), (8, 13), (14, 19), (20, 24), (24, 25), (0, 0)]


@pytest.mark.parametrize(
    ("focus", "window", "expected"),
    [
        ((8, 19), 0, [3, 4]),
        ((8, 19), 1, [2, 3, 4, 5]),
        ((8, 19), 9, [1, 2, 3, 4, 5, 6]),
        ((0, 1), 1, [1, 2]),
        ((10, 10), 0, [3]),
        ((14, 14), 0, [4]),
        ((25, 25), 1, [5, 6]),
    ],
)
def test_window_covers_focus_tokens_and_neighbours_but_never_special_tokens(
    focus: tuple[int, int], window: int, expected: list[int]
) -> None:
    assert window_positions(OFFSETS, focus_start=focus[0], focus_end=focus[1], window=window) == (
        expected
    )


def test_window_of_text_without_content_tokens_is_empty() -> None:
    assert window_positions([(0, 0), (0, 0)], focus_start=0, focus_end=0, window=3) == []


def test_crop_keeps_short_inputs_whole() -> None:
    assert crop([0, 5, 6, 2], [1, 2], max_length=4) == ([0, 5, 6, 2], [1, 2])


def test_crop_keeps_special_tokens_and_the_window() -> None:
    ids = [0, *range(10, 30), 2]

    cropped, positions = crop(ids, [18, 19], max_length=6)

    assert cropped == [0, 26, 27, 28, 29, 2]
    assert [cropped[p] for p in positions] == [27, 28]


def test_long_texts_are_scored_around_the_focus(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    text = "The dog sleeps in the garden every morning. " * 6 + "Their house is old."
    focus = Span(text.index("Their"), text.index("Their") + 5)

    [score] = make_scorer(tiny_tokenizer).score([ScoredText(text, focus)])

    assert math.isfinite(score)
    assert score < 0
