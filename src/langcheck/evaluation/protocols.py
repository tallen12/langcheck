"""Protocols of the evaluation domain."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.evaluation.m2 import GoldEdit
from langcheck.evaluation.models import ScoredEdit


class EditAnnotator(Protocol):
    """Finds and types edits between a tokenized source and a correction (ERRANT)."""

    def system_edits(self, source: Sequence[str], corrected: str) -> list[ScoredEdit]:
        """The edits that turn ``source`` into the untokenized ``corrected`` text."""
        ...

    def reference_edits(self, source: Sequence[str], reference: Sequence[str]) -> list[ScoredEdit]:
        """The edits that turn ``source`` into the tokenized ``reference``."""
        ...

    def gold_edits(self, source: Sequence[str], edits: Sequence[GoldEdit]) -> list[ScoredEdit]:
        """``edits`` from an M2 file, with their kinds."""
        ...
