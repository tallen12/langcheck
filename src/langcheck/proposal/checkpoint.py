"""Saving and loading a trained proposer. It needs the tagger it read while training."""

from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from langcheck.proposal.model import ProposerShape, TokenProposer, word_embeddings
from langcheck.tagging.model import SlotTagger

_WEIGHTS = "proposer.pt"
_INFO = "proposer.json"


class ProposerInfo(BaseModel):
    """What was trained, and on what."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tagger: str
    encoder_size: int
    embedding_size: int
    vocab_size: int
    size: int
    layers: int
    heads: int
    dropout: float
    train_sentences: int
    validation_sentences: int
    epoch: int
    validation_loss: float

    @classmethod
    def describing(
        cls,
        shape: ProposerShape,
        *,
        tagger: str,
        train_sentences: int,
        validation_sentences: int,
        epoch: int,
        validation_loss: float,
    ) -> ProposerInfo:
        """The info of a proposer of ``shape``."""
        return cls(
            tagger=tagger,
            encoder_size=shape.encoder_size,
            embedding_size=shape.embedding_size,
            vocab_size=shape.vocab_size,
            size=shape.size,
            layers=shape.layers,
            heads=shape.heads,
            dropout=shape.dropout,
            train_sentences=train_sentences,
            validation_sentences=validation_sentences,
            epoch=epoch,
            validation_loss=validation_loss,
        )


def shape_for(tagger: SlotTagger) -> ProposerShape:
    """The default shape of a proposer reading ``tagger``."""
    embeddings = word_embeddings(tagger)
    return ProposerShape(
        encoder_size=int(tagger.encoder.config.hidden_size),
        embedding_size=embeddings.size(1),
        vocab_size=embeddings.size(0),
    )


def save_proposer(directory: Path, proposer: TokenProposer, info: ProposerInfo) -> None:
    """Write everything :func:`load_proposer` needs into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(proposer.state_dict(), directory / _WEIGHTS)
    (directory / _INFO).write_text(info.model_dump_json(indent=2), encoding="utf-8")


def load_proposer(directory: Path, *, device: torch.device) -> tuple[TokenProposer, ProposerInfo]:
    """The proposer saved in ``directory``, on ``device``, ready to write."""
    info = ProposerInfo.model_validate_json((directory / _INFO).read_text(encoding="utf-8"))
    proposer = TokenProposer(
        ProposerShape(
            encoder_size=info.encoder_size,
            embedding_size=info.embedding_size,
            vocab_size=info.vocab_size,
            size=info.size,
            layers=info.layers,
            heads=info.heads,
            dropout=info.dropout,
        )
    )
    proposer.load_state_dict(
        torch.load(directory / _WEIGHTS, map_location="cpu", weights_only=True)
    )
    return proposer.to(device).eval(), info
