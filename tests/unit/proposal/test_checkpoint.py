from dataclasses import replace
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from langcheck.proposal.checkpoint import (
    ProposerInfo,
    load_proposer,
    save_proposer,
    shape_for,
)
from langcheck.proposal.model import TokenProposer
from tests.unit.tagging.test_model import tiny_tagger

CPU = torch.device("cpu")


def test_a_saved_proposer_loads_with_the_same_weights(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    tagger = tiny_tagger(tiny_tokenizer)
    shape = replace(shape_for(tagger), size=8, heads=2, layers=1)
    proposer = TokenProposer(shape)
    info = ProposerInfo.describing(
        shape,
        tagger="tagger",
        train_sentences=1,
        validation_sentences=1,
        epoch=1,
        validation_loss=0.5,
    )

    save_proposer(tmp_path / "out", proposer, info)
    loaded, read = load_proposer(tmp_path / "out", device=CPU)

    assert read == info
    assert not loaded.training
    for name, value in proposer.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
