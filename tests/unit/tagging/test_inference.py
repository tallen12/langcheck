import torch
from transformers import PreTrainedTokenizerFast

from langcheck.tagging.inference import TaggerScorer
from tests.unit.tagging.test_model import tiny_tagger


def make_scorer(tokenizer: PreTrainedTokenizerFast, *, batch_size: int, limit: int) -> TaggerScorer:
    return TaggerScorer(
        tiny_tagger(tokenizer),
        tokenizer,
        device=torch.device("cpu"),
        max_length=limit,
        batch_size=batch_size,
    )


def test_results_keep_the_order_and_do_not_depend_on_batching(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    sentences = [("The", "dog", "sleeps", "."), ("dog",), ("The", "dog")]

    one = make_scorer(tiny_tokenizer, batch_size=1, limit=32).score(sentences)
    many = make_scorer(tiny_tokenizer, batch_size=3, limit=32).score(sentences)

    assert [len(result.words) for result in many] == [4, 1, 2]
    assert [len(result.gaps) for result in many] == [5, 2, 3]
    for single, batched in zip(one, many, strict=True):
        for a, b in zip(single.words, batched.words, strict=True):
            assert abs(a - b) < 1e-4


def test_slots_past_the_length_limit_score_zero(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    [result] = make_scorer(tiny_tokenizer, batch_size=1, limit=6).score([("dog",) * 30])

    assert len(result.words) == 30
    assert len(result.gaps) == 31
    assert result.words[-1] == result.gaps[-1] == 0.0
    assert result.words[0] > 0.0


def test_context_changes_the_scores_of_the_same_words(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    scorer = make_scorer(tiny_tokenizer, batch_size=2, limit=32)
    words = ("The", "dog", ".")

    [plain] = scorer.score([words])
    [read] = scorer.score([words], contexts=["The cat sleeps."])

    assert len(read.words) == len(plain.words)
    assert read.words != plain.words
