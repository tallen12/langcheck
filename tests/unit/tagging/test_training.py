import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.tagging.inference import TaggerScorer
from langcheck.tagging.labels import SlotLabels
from langcheck.tagging.training import (
    HOLDOUT_BLOCK,
    HOLDOUT_EVERY,
    TrainingOptions,
    sentence_labels,
    split_holdout,
    train,
)
from tests.unit.tagging.test_model import tiny_tagger

WRONG = AnnotatedSentence(("The", "dog", "."), ((GoldEdit(1, 2, ("dogs",), "R:NOUN:NUM"),),))
RIGHT = AnnotatedSentence(("The", "cat", "."), ((),))
CPU = torch.device("cpu")


def test_holdout_takes_whole_blocks() -> None:
    sentences = [RIGHT] * (HOLDOUT_BLOCK * HOLDOUT_EVERY)

    training, held = split_holdout(sentences)

    assert len(held) == HOLDOUT_BLOCK
    assert len(training) == HOLDOUT_BLOCK * (HOLDOUT_EVERY - 1)


def test_a_slot_is_wrong_when_any_annotator_edited_it() -> None:
    sentence = AnnotatedSentence(
        ("a", "b"), ((GoldEdit(0, 1, (), "U:DET"),), (GoldEdit(2, 2, ("c",), "M:DET"),))
    )

    assert sentence_labels(sentence) == SlotLabels((True, False), (False, False, True))


def test_training_learns_a_word_and_chooses_a_threshold(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    lines: list[str] = []
    sentences = [WRONG, RIGHT] * 8

    result = train(
        tagger,
        tiny_tokenizer,
        sentences,
        [WRONG, RIGHT],
        device=CPU,
        options=TrainingOptions(epochs=12, batch_size=4, learning_rate=3e-3, warmup_share=0.0),
        log=lines.append,
    )

    scorer = TaggerScorer(tagger, tiny_tokenizer, device=CPU, max_length=32)
    [wrong, right] = scorer.score([WRONG.tokens, RIGHT.tokens])
    assert wrong.words[1] > right.words[1]
    assert 1 <= result.epoch <= 12
    assert result.validation.slots.tp == 1
    assert lines[0].startswith("epoch 1:")


def test_training_needs_validation_sentences(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="no validation"):
        train(
            tiny_tagger(tiny_tokenizer),
            tiny_tokenizer,
            [RIGHT],
            [],
            device=CPU,
            options=TrainingOptions(),
            log=lambda _: None,
        )


def test_training_needs_an_epoch(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="epochs"):
        train(
            tiny_tagger(tiny_tokenizer),
            tiny_tokenizer,
            [RIGHT],
            [RIGHT],
            device=CPU,
            options=TrainingOptions(epochs=0),
            log=lambda _: None,
        )
