"""Clean references from the panel's consensus: gold without the edits judges reject.

A sentence keeps only the edits the panel agrees are valid. Each version of it (an
annotator's correction) becomes one reference with its valid edits; the preferred version
comes first as the main reference. A sentence the panel has not settled, or that still
has an error no pooled edit fixes, is left out with the reason, since scoring against it
would count a right fix as a false alarm or a missed one.

Linked (non-local) edits only work together with another edit, which the local proposer
cannot make. In the local scope a sentence that needs one is left out; in the full scope
it is kept and marked, so a future non-local model can be scored and trained on it.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from langcheck.judging.consensus import Agreed, Disputed, Pending
from langcheck.judging.models import EditKey, SentenceKey, Verdict, Version
from langcheck.judging.panel import ItemConsensus
from langcheck.judging.render import render


class Scope(StrEnum):
    """Which fixes a clean set keeps."""

    LOCAL = "local"
    """Only sentences whose valid edits each work alone, for a local model."""
    ALL = "all"
    """Linked edits too, with the sentence marked."""


class LeftOutReason(StrEnum):
    """Why a sentence is not in a clean set."""

    PENDING = "pending"
    """A judge has not answered on an edit or the sentence."""
    DISPUTED = "disputed"
    """The judges split on an edit or on whether an error is left; a person decides."""
    ERROR_LEFT = "error_left"
    """An error is left that no pooled edit fixes."""
    NON_LOCAL = "non_local"
    """A valid edit is linked to another, outside the local scope."""


@dataclass(frozen=True, slots=True)
class CleanSentence:
    """A sentence as written and its references, the main one first."""

    sentence: SentenceKey
    text: str
    references: tuple[str, ...]
    linked: bool
    """Whether a reference needs a linked (non-local) edit."""


@dataclass(frozen=True, slots=True)
class LeftOut:
    """A sentence a clean set leaves out, and why."""

    sentence: SentenceKey
    reason: LeftOutReason


type Cleaned = CleanSentence | LeftOut


class CleanLine(BaseModel):
    """One line of a clean set; ``text`` and ``corrected`` read as a corrections file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: SentenceKey
    text: str
    corrected: str
    alternatives: tuple[str, ...] = ()
    linked: bool = False


def clean(decided: ItemConsensus, *, scope: Scope) -> Cleaned:
    """``decided``'s sentence with its valid edits as references, or why it is left out."""
    item = decided.item
    outcomes = [*decided.edits.values(), decided.remaining]
    if any(isinstance(outcome, Pending) for outcome in outcomes):
        return LeftOut(item.key, LeftOutReason.PENDING)
    if any(isinstance(outcome, Disputed) for outcome in outcomes):
        return LeftOut(item.key, LeftOutReason.DISPUTED)
    if isinstance(decided.remaining, Agreed) and decided.remaining.value:
        return LeftOut(item.key, LeftOutReason.ERROR_LEFT)
    valid = _valid(decided)
    linked = bool(valid & decided.linked)
    match scope:
        case Scope.LOCAL if linked:
            return LeftOut(item.key, LeftOutReason.NON_LOCAL)
        case Scope.LOCAL | Scope.ALL:
            pass
        case unreachable:
            assert_never(unreachable)
    references = [
        tuple(edit for edit in version.edits if edit in valid)
        for version in _ranked(decided, valid)
    ]
    # A version whose edits were all rejected would let a checker that changes nothing
    # match it while another version fixes a real error.
    kept = list(dict.fromkeys(edits for edits in references if edits)) or [()]
    return CleanSentence(
        item.key,
        render(item.tokens, (), item.written),
        tuple(render(item.tokens, edits, item.written) for edits in kept),
        linked,
    )


def clean_line(sentence: CleanSentence) -> CleanLine:
    """``sentence`` as a line of a clean set."""
    main, *others = sentence.references
    return CleanLine(
        sentence=sentence.sentence,
        text=sentence.text,
        corrected=main,
        alternatives=tuple(others),
        linked=sentence.linked,
    )


def _valid(decided: ItemConsensus) -> set[EditKey]:
    return {
        key
        for key, outcome in decided.edits.items()
        if isinstance(outcome, Agreed) and outcome.value is Verdict.VALID
    }


def _ranked(decided: ItemConsensus, valid: set[EditKey]) -> list[Version]:
    """The versions, the preferred one first, else the one with most valid edits first."""
    versions = list(decided.item.versions)
    match decided.preference:
        case Agreed(value=label) if any(v.label == label for v in versions):
            return sorted(versions, key=lambda v: v.label != label)
        case _:
            return sorted(versions, key=lambda v: -sum(edit in valid for edit in v.edits))
