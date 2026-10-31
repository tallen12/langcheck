import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast

from langcheck.tagging.checkpoint import CheckpointInfo, save_checkpoint
from langcheck.tagging.loaded import (
    OneAtATimeCommas,
    OneAtATimeScorer,
    load_trained,
    torch_limit,
)
from tests.conftest import WordTableTagger
from tests.unit.tagging.test_model import tiny_tagger


class LockedOut:
    """Scores only while the lock is held, so a test can tell the wrapper took it."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[tuple[float, ...]]:
        assert self._lock.locked()
        return [(0.5,) for _ in sentences]


def test_the_scorer_runs_holding_the_lock_and_lets_go() -> None:
    lock = threading.Lock()
    scorer = OneAtATimeScorer(WordTableTagger({"go": 0.9}), lock)

    [scores] = scorer.score([("go",)])

    assert scores.words == (0.9,)
    assert not lock.locked()


def test_the_comma_scorer_shares_the_lock() -> None:
    lock = threading.Lock()

    assert OneAtATimeCommas(LockedOut(lock), lock).score([("go",)]) == [(0.5,)]
    assert not lock.locked()


def test_a_saved_tagger_loads_ready_to_score(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    info = CheckpointInfo(
        base_model="tiny",
        threshold=0.3,
        precision_target=0.5,
        max_length=32,
        train_sentences=2,
        validation_sentences=1,
        epoch=1,
        context_sentences=2,
    )
    save_checkpoint(tmp_path, tiny_tagger(tiny_tokenizer), tiny_tokenizer, info)

    trained = load_trained(tmp_path, device="cpu")

    assert (trained.threshold, trained.context_sentences) == (0.3, 2)
    assert trained.commas is None
    assert len(trained.scorer.score([("The", "dog")])) == 1


def test_a_missing_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_trained(tmp_path / "missing", device="cpu")


def test_the_limit_ignores_anything_but_a_cuda_device() -> None:
    torch_limit("cpu")
