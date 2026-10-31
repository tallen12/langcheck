"""Loading a saved tagger, for whichever entry point needs one. PyTorch is imported only
when a tagger is actually loaded.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langcheck.tagging.models import SentenceReading, SlotScores
from langcheck.tagging.protocols import GapCommaScorer, SentenceReader, SlotScorer
from langcheck.tagging.reading import PairedReader

MAX_VRAM_SHARE = 0.8


@dataclass(frozen=True, slots=True)
class Trained:
    """A loaded tagger and what evaluating it needs to know."""

    scorer: SlotScorer
    threshold: float
    base_model: str
    precision_target: float
    commas: GapCommaScorer | None = None
    context_sentences: int = 0
    """``None`` until ``langcheck tagger train-commas`` has run on the checkpoint."""
    shared_pass: SentenceReader | None = None

    @property
    def reader(self) -> SentenceReader:
        """What flagging reads with: one pass of the model when it has that, or else the
        scorers one after the other.
        """
        return self.shared_pass or PairedReader(self.scorer, self.commas)


class OneAtATimeScorer:
    """A ``SlotScorer`` that waits its turn: a server checks several documents at once, and
    every batch at once could run out of memory.
    """

    def __init__(self, scorer: SlotScorer, lock: threading.Lock) -> None:
        self._scorer = scorer
        self._lock = lock

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        """The wrapped scorer's scores, once no other batch is running. Blocking."""
        with self._lock:
            return self._scorer.score(sentences, contexts=contexts)


class OneAtATimeCommas:
    """A ``GapCommaScorer`` sharing the tagger's turn, since it runs the same model."""

    def __init__(self, scorer: GapCommaScorer, lock: threading.Lock) -> None:
        self._scorer = scorer
        self._lock = lock

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[tuple[float, ...]]:
        """The wrapped scorer's scores, once no other batch is running. Blocking."""
        with self._lock:
            return self._scorer.score(sentences, contexts=contexts)


class OneAtATimeReader:
    """A ``SentenceReader`` sharing the same turn."""

    def __init__(self, reader: SentenceReader, lock: threading.Lock) -> None:
        self._reader = reader
        self._lock = lock

    def read(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SentenceReading]:
        """The wrapped reader's readings, once no other batch is running. Blocking."""
        with self._lock:
            return self._reader.read(sentences, contexts=contexts)


def torch_limit(device: object) -> None:
    """Past VRAM the driver spills into system RAM, which can take the desktop down."""
    import torch

    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(MAX_VRAM_SHARE)


def load_trained(path: Path, *, device: str) -> Trained:
    """The tagger saved at ``path``, safe to call from several threads. Blocking."""
    import torch

    from langcheck.tagging.checkpoint import load_checkpoint
    from langcheck.tagging.comma import CommaScorer, has_comma_head, load_comma_head
    from langcheck.tagging.inference import TaggerReader, TaggerScorer

    target = torch.device(device)
    torch_limit(target)
    loaded = load_checkpoint(path, device=target)
    length = loaded.info.max_length
    lock = threading.Lock()
    commas = None
    head = None
    if has_comma_head(path):
        head = load_comma_head(path, loaded.tagger, device=target)
        commas = OneAtATimeCommas(
            CommaScorer(loaded.tagger, head, loaded.tokenizer, device=target, max_length=length),
            lock,
        )
    return Trained(
        OneAtATimeScorer(
            TaggerScorer(loaded.tagger, loaded.tokenizer, device=target, max_length=length), lock
        ),
        loaded.info.threshold,
        loaded.info.base_model,
        loaded.info.precision_target,
        commas,
        loaded.info.context_sentences,
        OneAtATimeReader(
            TaggerReader(
                loaded.tagger,
                loaded.tokenizer,
                device=target,
                max_length=length,
                comma_head=head,
            ),
            lock,
        ),
    )
