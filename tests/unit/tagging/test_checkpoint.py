from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from transformers import PreTrainedTokenizerFast

from langcheck.tagging.checkpoint import CheckpointInfo, load_checkpoint, save_checkpoint
from langcheck.tagging.inference import TaggerScorer
from tests.unit.tagging.test_model import tiny_tagger

INFO = CheckpointInfo(
    base_model="tiny",
    threshold=0.3,
    precision_target=0.5,
    max_length=32,
    train_sentences=2,
    validation_sentences=1,
    epoch=1,
)
CPU = torch.device("cpu")


def test_a_saved_tagger_scores_the_same_when_loaded(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    words = ("The", "dog", ".")
    before = TaggerScorer(tagger, tiny_tokenizer, device=CPU, max_length=32).score([words])

    save_checkpoint(tmp_path, tagger, tiny_tokenizer, INFO)
    loaded = load_checkpoint(tmp_path, device=CPU)
    after = TaggerScorer(loaded.tagger, loaded.tokenizer, device=CPU, max_length=32).score([words])

    assert loaded.info == INFO
    assert after[0].words == pytest.approx(before[0].words)


def test_a_damaged_info_file_is_refused(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    save_checkpoint(tmp_path, tiny_tagger(tiny_tokenizer), tiny_tokenizer, INFO)
    (tmp_path / "tagger.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_checkpoint(tmp_path, device=CPU)
