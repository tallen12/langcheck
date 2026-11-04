import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.tagging.flag_reach import FlagReach
from langcheck.tagging.inference import TaggerScorer
from langcheck.tagging.labels import SlotLabels
from langcheck.tagging.selection import ByFlags
from langcheck.tagging.tiers import TierPolicy
from langcheck.tagging.training import (
    HOLDOUT_BLOCK,
    HOLDOUT_EVERY,
    InContext,
    TrainingOptions,
    flag_line,
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
    sentences = [InContext(WRONG), InContext(RIGHT)] * 8

    result = train(
        tagger,
        tiny_tokenizer,
        sentences,
        [InContext(WRONG), InContext(RIGHT)],
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
            [InContext(RIGHT)],
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
            [InContext(RIGHT)],
            [InContext(RIGHT)],
            device=CPU,
            options=TrainingOptions(epochs=0),
            log=lambda _: None,
        )


def test_judging_epochs_by_flags_logs_them_and_keeps_the_best(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    lines: list[str] = []

    result = train(
        tiny_tagger(tiny_tokenizer),
        tiny_tokenizer,
        [InContext(WRONG), InContext(RIGHT)] * 8,
        [InContext(WRONG), InContext(RIGHT)],
        device=CPU,
        options=TrainingOptions(
            epochs=6,
            batch_size=4,
            learning_rate=3e-3,
            warmup_share=0.0,
            choice=ByFlags(TierPolicy()),
        ),
        log=lines.append,
    )

    flagged = [line for line in lines if "edits inside a flag" in line]
    assert [line.split(":")[0] for line in flagged] == [f"epoch {e}" for e in range(1, 7)]
    assert result.flags is not None
    best = max(float(line.rsplit(" ", 1)[1]) for line in flagged)
    assert result.flags.f05 == pytest.approx(best, abs=1e-4)
    assert lines[-1] == f"keeping epoch {result.epoch}"


def test_flag_line_shows_precision_reach_and_f05() -> None:
    line = flag_line(2, TierPolicy(), FlagReach(flags=10, right=8, edits=12, reached=9))

    assert line == (
        "epoch 2: flags from 0.30, 10 flags, precision 0.800, 9/12 edits inside a flag "
        "(reach 0.750), F0.5 0.7895"
    )
