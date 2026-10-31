"""Value types for correcting flags: the policy, the candidates and how each one fared."""

from dataclasses import dataclass, field
from enum import StrEnum

from langcheck.check.resolve import ValidationPolicy
from langcheck.check.suggestion import Confidence
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.tiers import FlagKind, TextFlag
from langcheck.taxonomy.kinds import ErrorKind, Operation


class FlagType(StrEnum):
    """What the LLM is told about a flag. The tagger only knows where an error is and whether
    something is missing there, not which grammatical kind of error it is.
    """

    WRONG_TEXT = "wrong-text"
    MISSING_TEXT = "missing-text"
    OPTIONAL_COMMA = "optional-comma"


def flag_type(flag: TextFlag) -> FlagType:
    """The type of ``flag``."""
    if flag.kind is FlagKind.OPTIONAL_COMMA:
        return FlagType.OPTIONAL_COMMA
    return FlagType.MISSING_TEXT if flag.span.start == flag.span.end else FlagType.WRONG_TEXT


class RejectionReason(StrEnum):
    """Why a correction from the LLM is not offered."""

    UNKNOWN_KIND = "unknown_kind"
    NOT_AT_FLAG = "not_at_flag"
    NO_CHANGE = "no_change"
    TOO_LARGE = "too_large"
    EMPTIES_SENTENCE = "empties_sentence"
    DUPLICATE = "duplicate"
    NEW_ERROR = "new_error"
    STILL_FLAGGED = "still_flagged"
    NO_IMPROVEMENT = "no_improvement"


@dataclass(frozen=True, slots=True)
class CorrectionPolicy:
    """How many corrections to ask for per flag and what makes one worth offering.

    ``min_reduction`` is how much a correction must lower its sentence's score to be
    offered, and ``confident_reduction`` to be offered with high confidence. ``edit``
    limits the size of an edit: deleting words always lowers the score, so without a limit
    the tagger would reward cutting a sentence down.
    """

    count: int = 2
    min_reduction: float = 0.1
    confident_reduction: float = 0.6
    edit: ValidationPolicy = field(default_factory=ValidationPolicy)

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"count must be at least 1, got {self.count}")
        if self.min_reduction < 0:
            raise ValueError(f"min_reduction must be non-negative, got {self.min_reduction}")
        if self.confident_reduction < self.min_reduction:
            msg = (
                f"confident_reduction must be at least min_reduction, "
                f"got {self.confident_reduction} and {self.min_reduction}"
            )
            raise ValueError(msg)

    def confidence(self, reduction: float) -> Confidence:
        """The confidence of an offered correction lowering the score by ``reduction``."""
        return Confidence.HIGH if reduction >= self.confident_reduction else Confidence.LOW


@dataclass(frozen=True, slots=True)
class FlaggedSentence:
    """A sentence with at least one flag. Flags have offsets into the whole text."""

    sentence: SentenceInText
    flags: tuple[TextFlag, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A correction placed in the whole text: replace ``span`` with ``replacement``."""

    span: CharSpan
    original: str
    replacement: str
    kind: ErrorKind
    operation: Operation
    explanation: str
    position: int = 0
    """Where the LLM put it among the flag's corrections, best first."""


@dataclass(frozen=True, slots=True)
class Ranked:
    """A correction worth offering, its sentence's score before and after it, and how
    prominently to show it.
    """

    candidate: Candidate
    score_before: float
    score_after: float
    confidence: Confidence

    @property
    def reduction(self) -> float:
        """How much the correction lowers its sentence's score."""
        return self.score_before - self.score_after


@dataclass(frozen=True, slots=True)
class Rejected:
    """A correction not offered. ``reduction`` is ``None`` when it never reached the tagger;
    ``candidate`` is ``None`` when it could not be applied, so measuring cannot judge it.
    """

    original: str
    replacement: str
    reason: RejectionReason
    reduction: float | None = None
    candidate: Candidate | None = None


@dataclass(frozen=True, slots=True)
class Pending:
    """A flag's candidates, waiting for the tagger, and those already rejected."""

    flagged: FlaggedSentence
    flag: TextFlag
    candidates: tuple[Candidate, ...]
    rejected: tuple[Rejected, ...]


@dataclass(frozen=True, slots=True)
class Corrected:
    """A flag with an answer from the LLM: corrections by score reduction, largest first.
    Both are empty when the LLM judged the flagged text correct.
    """

    flag: TextFlag
    ranked: tuple[Ranked, ...]
    rejected: tuple[Rejected, ...]


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A flag the LLM gave no valid reply for: unknown, not a false alarm."""

    flag: TextFlag
    reason: str


type FlagOutcome = Corrected | Unavailable
