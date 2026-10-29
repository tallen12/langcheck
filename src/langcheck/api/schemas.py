"""Request and response bodies of the HTTP API, and conversion from domain values.

Offsets are Unicode code points of the text the client sent.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from langcheck.check.checker import CheckResult, Unchecked
from langcheck.check.suggestion import Suggestion
from langcheck.context.models import DocumentContext, Register
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
    glossary: list[str]
    notes: str

    @classmethod
    def of(cls, context: DocumentContext) -> ContextBody:
        """The body for ``context``."""
        return cls(
            summary=context.summary,
            register=context.register,
            dialect=context.dialect,
            glossary=list(context.glossary),
            notes=context.notes,
        )

    def to_context(self) -> DocumentContext:
        """The domain value."""
        return DocumentContext(
            summary=self.summary,
            register=self.writing_register,
            dialect=self.dialect,
            glossary=tuple(self.glossary),
            notes=self.notes,
        )


class CheckRequest(BaseModel):
    """A one-shot check. Without ``context`` the server builds one from the text."""

    model_config = _BODY

    text: Text
    dictionary: Words = []
    context: ContextBody | None = None


class ContextRequest(BaseModel):
    """Ask for the context of a text."""

    model_config = _BODY

    text: Text
    dictionary: Words = []


class OptionBody(BaseModel):
    """One candidate replacement, best first."""

    model_config = _BODY

    replacement: str


class SuggestionBody(BaseModel):
    """One suggested fix for ``text[start:end]``."""

    model_config = _BODY

    id: str
    start: int
    end: int
    original: str
    kind: ErrorKind
    operation: Operation
    explanation: str
    options: list[OptionBody]

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
            options=[OptionBody(replacement=o.replacement) for o in suggestion.options],
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
