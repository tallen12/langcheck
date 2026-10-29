import math

import pytest
import torch
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
)

from langcheck.scoring.cola import AcceptabilityScorer
from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import SentenceScorer
from langcheck.text.models import Span


def make_scorer(
    tokenizer: PreTrainedTokenizerFast, *, batch_size: int = 32, label: int = 1
) -> AcceptabilityScorer:
    """A random two-layer RoBERTa classifier, seeded so scores repeat."""
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
    return AcceptabilityScorer(
        RobertaForSequenceClassification(config),
        tokenizer,
        device="cpu",
        acceptable_label=label,
        batch_size=batch_size,
        max_length=32,
    )


TEXTS = [
    ScoredText("I think their going home.", Span(8, 13)),
    ScoredText("I think they're going home.", Span(8, 15)),
    ScoredText("The dog sleeps.", Span(0, 3)),
]


def test_scores_are_log_probabilities_of_the_acceptable_class(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    scorer: SentenceScorer = make_scorer(tiny_tokenizer)
    other_label = make_scorer(tiny_tokenizer, label=0)

    acceptable = scorer.score(TEXTS)
    unacceptable = other_label.score(TEXTS)

    for yes, no in zip(acceptable, unacceptable, strict=True):
        assert yes <= 0
        assert math.exp(yes) + math.exp(no) == pytest.approx(1.0, abs=1e-5)


def test_batching_and_padding_do_not_change_scores(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    together = make_scorer(tiny_tokenizer).score(TEXTS)
    one_by_one = make_scorer(tiny_tokenizer, batch_size=1).score(TEXTS)

    assert one_by_one == pytest.approx(together, abs=1e-5)
    assert make_scorer(tiny_tokenizer).score([]) == []


def test_rejects_a_bad_batch_size(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        make_scorer(tiny_tokenizer, batch_size=0)
