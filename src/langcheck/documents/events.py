"""What clients watching a document are told."""

from dataclasses import dataclass

from langcheck.check.checker import Unchecked
from langcheck.check.suggestion import Suggestion
from langcheck.documents.models import CheckState, Document


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The current version's suggestions and check state. Clients replace what they
    show with each snapshot, so a missed event never leaves stale suggestions behind.
    """

    version: int
    state: CheckState
    suggestions: tuple[Suggestion, ...]
    unchecked: tuple[Unchecked, ...]

    @classmethod
    def of(cls, document: Document) -> Snapshot:
        """The snapshot of ``document``."""
        return cls(document.version, document.state, document.suggestions, document.unchecked)


@dataclass(frozen=True, slots=True)
class Closed:
    """The document was closed or renamed; no more events follow."""


type DocumentEvent = Snapshot | Closed
