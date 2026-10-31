from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.tagging.comma import (
    CommaScorer,
    gap_labels,
    has_comma_head,
    load_comma_head,
    save_comma_head,
    train_comma_head,
)
from tests.unit.tagging.test_model import tiny_tagger

CPU = torch.device("cpu")
ADD_COMMA = AnnotatedSentence(("The", "dog", "."), ((GoldEdit(1, 1, (",",), "M:PUNCT"),),))
ADD_THE = AnnotatedSentence(("dog", "cat", "."), ((GoldEdit(1, 1, ("the",), "M:DET"),),))


def test_labels_cover_only_gaps_with_insertions() -> None:
    sentence = AnnotatedSentence(
        ("a", "b"),
        ((GoldEdit(0, 1, ("c",), "R:OTHER"), GoldEdit(1, 1, (",",), "M:PUNCT")),),
    )

    assert gap_labels(sentence) == {1: True}


def test_annotators_who_disagree_on_a_gap_make_it_no_comma() -> None:
    sentence = AnnotatedSentence(
        ("a",), ((GoldEdit(1, 1, (",",), "M"),), (GoldEdit(1, 1, ("the",), "M"),))
    )

    assert gap_labels(sentence) == {1: False}


def test_the_head_learns_which_gaps_lack_commas_and_leaves_the_tagger_alone(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    before = {name: value.clone() for name, value in tagger.state_dict().items()}
    lines: list[str] = []

    head = train_comma_head(
        tagger,
        tiny_tokenizer,
        [ADD_COMMA, ADD_THE] * 8,
        device=CPU,
        max_length=32,
        log=lines.append,
    )
    scorer = CommaScorer(tagger, head, tiny_tokenizer, device=CPU, max_length=32)
    [comma, other] = scorer.score([ADD_COMMA.tokens, ADD_THE.tokens])

    assert comma[1] > other[1]
    assert len(comma) == 4
    assert lines[0] == "16 gold insertions, 8 of them commas"
    for name, value in tagger.state_dict().items():
        assert torch.equal(value, before[name])


def test_nothing_to_train_on_is_refused(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="no gold insertions"):
        train_comma_head(
            tiny_tagger(tiny_tokenizer),
            tiny_tokenizer,
            [AnnotatedSentence(("a",), ((),))],
            device=CPU,
            max_length=32,
            log=lambda _: None,
        )


def test_gaps_past_the_length_limit_score_zero(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    head = torch.nn.Linear(tagger.gap_state_size, 1)
    scorer = CommaScorer(tagger, head, tiny_tokenizer, device=CPU, max_length=6)

    [scores] = scorer.score([("dog",) * 30])

    assert len(scores) == 31
    assert scores[-1] == 0.0


def test_a_saved_head_loads_the_same(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    head = torch.nn.Linear(tagger.gap_state_size, 1)
    assert not has_comma_head(tmp_path)

    save_comma_head(tmp_path, head)

    assert has_comma_head(tmp_path)
    loaded = load_comma_head(tmp_path, tagger, device=CPU)
    assert torch.equal(loaded.weight, head.weight)
