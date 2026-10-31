"""Request and response bodies of the HTTP API, and conversion from domain values.

Offsets are Unicode code points of the text the client sent.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from langcheck.check.checker import CheckResult, Unchecked
from langcheck.check.suggestion import Confidence, Option, Scored, Suggestion, Unscored
from langcheck.context.models import DocumentContext, Register
from langcheck.documents.events import Snapshot
from langcheck.documents.models import CheckState
from langcheck.taxonomy.kinds import ErrorKind, Operation

MAX_TEXT_LENGTH = 500_000
MAX_DICTIONARY_WORDS = 10_000

_BODY = ConfigDict(frozen=True, extra="forbid")

type Text = Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]
type Words = Annotated[
    list[Annotated[str, Field(max_length=200)]], Field(max_length=MAX_DICTIONARY_WORDS)
]


class ContextBody(BaseModel):
    """A document context, sent back by ``/context`` and accepted by ``/check``."""

    model_config = _BODY

    summary: str
    # Aliased because ``register`` would shadow ``BaseModel.register``.
    writing_register: Annotated[Register, Field(alias="register")]
    dialect: str
    notes: str

    @classmethod
    def of(cls, context: DocumentContext) -> ContextBody:
        """The body for ``context``."""
        return cls(
            summary=context.summary,
            register=context.register,
            dialect=context.dialect,
            notes=context.notes,
        )

    def to_context(self) -> DocumentContext:
        """The domain value."""
        return DocumentContext(
            summary=self.summary,
            register=self.writing_register,
            dialect=self.dialect,
            notes=self.notes,
        )


class CheckRequest(BaseModel):
    """A one-shot check. Without ``context`` the server builds one from the text. With
    ``uri``, that document's stored dictionary is used as well as ``dictionary``.
    """

    model_config = _BODY

    text: Text
    dictionary: Words = []
    context: ContextBody | None = None
    uri: Annotated[str | None, Field(max_length=2000)] = None


class ContextRequest(BaseModel):
    """Ask for the context of a text."""

    model_config = _BODY

    text: Text


class OptionBody(BaseModel):
    """One candidate replacement, best first. ``score`` and ``delta`` (how much the
    candidate sentence improves on the original) are null when no scorer ran. ``confidence``
    says how prominently to show it.
    """

    model_config = _BODY

    replacement: str
    score: float | None = None
    delta: float | None = None
    confidence: Confidence = Confidence.HIGH

    @classmethod
    def of(cls, option: Option) -> OptionBody:
        """The body for ``option``."""
        confidence = option.confidence
        match option.rating:
            case Scored(score=score, delta=delta):
                return cls(
                    replacement=option.replacement, score=score, delta=delta, confidence=confidence
                )
            case Unscored():
                return cls(replacement=option.replacement, confidence=confidence)


class SuggestionBody(BaseModel):
    """One suggested fix for ``text[start:end]``; ``confidence`` is its best option's."""

    model_config = _BODY

    id: str
    start: int
    end: int
    original: str
    kind: ErrorKind
    operation: Operation
    explanation: str
    options: list[OptionBody]
    confidence: Confidence = Confidence.HIGH

    @classmethod
    def of(cls, suggestion: Suggestion) -> SuggestionBody:
        """The body for ``suggestion``."""
        return cls(
            id=suggestion.id,
            start=suggestion.span.start,
            end=suggestion.span.end,
            original=suggestion.original,
            kind=suggestion.kind,
            operation=suggestion.operation,
            explanation=suggestion.explanation,
            options=[OptionBody.of(o) for o in suggestion.options],
            confidence=suggestion.confidence,
        )


class UncheckedBody(BaseModel):
    """A span the LLM gave no valid answer for: unknown, not clean."""

    model_config = _BODY

    start: int
    end: int
    reason: str

    @classmethod
    def of(cls, unchecked: Unchecked) -> UncheckedBody:
        """The body for ``unchecked``."""
        return cls(start=unchecked.span.start, end=unchecked.span.end, reason=unchecked.reason)


class CheckResponse(BaseModel):
    """The result of ``POST /check``, with the context the check used."""

    model_config = _BODY

    suggestions: list[SuggestionBody]
    unchecked: list[UncheckedBody]
    context: ContextBody

    @classmethod
    def of(cls, result: CheckResult, context: DocumentContext) -> CheckResponse:
        """The body for ``result``."""
        return cls(
            suggestions=[SuggestionBody.of(s) for s in result.suggestions],
            unchecked=[UncheckedBody.of(u) for u in result.unchecked],
            context=ContextBody.of(context),
        )


class HealthResponse(BaseModel):
    """``GET /health``."""

    model_config = _BODY

    status: str = "ok"


class UpdateRequest(BaseModel):
    """The full text of a new version; ``base_version`` guards against lost updates."""

    model_config = _BODY

    text: Text
    base_version: Annotated[int | None, Field(ge=0)] = None


class VersionBody(BaseModel):
    """The document's version after a change."""

    model_config = _BODY

    version: int


class Conflict(BaseModel):
    """What conflicted, and the ``current`` version."""

    model_config = _BODY

    message: str
    current: int


class ConflictBody(BaseModel):
    """A 409 reply: the request named a version (or URI) that is not current."""

    model_config = _BODY

    detail: Conflict


class RenameRequest(BaseModel):
    """The document's new URI."""

    model_config = _BODY

    uri: Annotated[str, Field(min_length=1, max_length=2000)]


class AcceptRequest(BaseModel):
    """Which option to apply, and the version the client saw it in."""

    model_config = _BODY

    option: Annotated[int, Field(ge=0)] = 0
    base_version: Annotated[int | None, Field(ge=0)] = None


class AcceptBody(BaseModel):
    """The edit to make: ``[start, end)`` of the previous version becomes
    ``replacement``, giving ``version``.
    """

    model_config = _BODY

    version: int
    start: int
    end: int
    replacement: str


class DocumentBody(BaseModel):
    """The current version's suggestions and check state."""

    model_config = _BODY

    version: int
    state: CheckState
    suggestions: list[SuggestionBody]
    unchecked: list[UncheckedBody]

    @classmethod
    def of(cls, snapshot: Snapshot) -> DocumentBody:
        """The body for ``snapshot``."""
        return cls(
            version=snapshot.version,
            state=snapshot.state,
            suggestions=[SuggestionBody.of(s) for s in snapshot.suggestions],
            unchecked=[UncheckedBody.of(u) for u in snapshot.unchecked],
        )


class WordsRequest(BaseModel):
    """Words to add to or remove from a document's dictionary."""

    model_config = _BODY

    words: Annotated[Words, Field(min_length=1)]


class WordsBody(BaseModel):
    """A document's dictionary, sorted."""

    model_config = _BODY

    words: list[str]
