"""Saving and loading a trained tagger: weights, tokenizer and operating threshold."""

from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerBase

from langcheck.tagging.encoding import load_tokenizer
from langcheck.tagging.model import SlotTagger

_WEIGHTS = "tagger.pt"
_INFO = "tagger.json"


class CheckpointInfo(BaseModel):
    """What was trained, and the threshold validation chose for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str
    threshold: float
    precision_target: float
    max_length: int
    train_sentences: int
    validation_sentences: int
    epoch: int


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A loaded tagger, ready to score."""

    tagger: SlotTagger
    tokenizer: PreTrainedTokenizerBase
    info: CheckpointInfo


def save_checkpoint(
    directory: Path,
    tagger: SlotTagger,
    tokenizer: PreTrainedTokenizerBase,
    info: CheckpointInfo,
) -> None:
    """Write everything :func:`load_checkpoint` needs into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    tagger.encoder.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save(tagger.state_dict(), directory / _WEIGHTS)
    (directory / _INFO).write_text(info.model_dump_json(indent=2), encoding="utf-8")


def load_checkpoint(directory: Path, *, device: torch.device) -> Checkpoint:
    """The tagger saved in ``directory``, on ``device``."""
    info = CheckpointInfo.model_validate_json((directory / _INFO).read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(directory)
    tagger = SlotTagger(AutoModel.from_config(config))
    tagger.load_state_dict(torch.load(directory / _WEIGHTS, map_location="cpu", weights_only=True))
    tagger.to(device)
    return Checkpoint(tagger.eval(), load_tokenizer(directory), info)
