import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.tagging.comma import CommaScorer
from langcheck.tagging.inference import TaggerReader, TaggerScorer
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


def test_one_pass_reads_what_the_separate_scorers_read(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    head = torch.nn.Linear(tagger.gap_state_size, 1)
    cpu = torch.device("cpu")
    sentences = [("The", "dog", "sleeps", "."), ("dog",)]
    contexts = ["The cat sleeps.", ""]

    readings = TaggerReader(
        tagger, tiny_tokenizer, device=cpu, max_length=32, comma_head=head
    ).read(sentences, contexts=contexts)
    scores = TaggerScorer(tagger, tiny_tokenizer, device=cpu, max_length=32).score(
        sentences, contexts=contexts
    )
    commas = CommaScorer(tagger, head, tiny_tokenizer, device=cpu, max_length=32).score(
        sentences, contexts=contexts
    )

    for reading, score, comma in zip(readings, scores, commas, strict=True):
        assert reading.scores.words == pytest.approx(score.words, abs=1e-4)
        assert reading.scores.gaps == pytest.approx(score.gaps, abs=1e-4)
        assert reading.commas == pytest.approx(comma, abs=1e-4)


def test_one_pass_runs_the_encoder_once_per_batch(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    head = torch.nn.Linear(tagger.gap_state_size, 1)
    calls: list[int] = []
    tagger.encoder.register_forward_hook(lambda *_: calls.append(1))

    TaggerReader(
        tagger, tiny_tokenizer, device=torch.device("cpu"), max_length=32, comma_head=head
    ).read([("The", "dog")])

    assert len(calls) == 1


def test_commas_past_the_length_limit_score_zero(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    head = torch.nn.Linear(tagger.gap_state_size, 1)

    [reading] = TaggerReader(
        tagger, tiny_tokenizer, device=torch.device("cpu"), max_length=6, comma_head=head
    ).read([("dog",) * 30])

    assert reading.commas is not None
    assert len(reading.commas) == 31
    assert reading.commas[-1] == 0.0


def test_without_a_comma_head_a_reading_has_no_commas(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    [reading] = TaggerReader(
        tiny_tagger(tiny_tokenizer), tiny_tokenizer, device=torch.device("cpu"), max_length=32
    ).read([("dog",)])

    assert reading.commas is None
