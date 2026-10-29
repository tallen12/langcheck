"""The service's JSON, parsed by hand into small frozen types (no Pydantic here)."""

from __future__ import annotations

from dataclasses import dataclass


class SchemaError(ValueError):
    """The service sent JSON that does not have the expected shape."""


@dataclass(frozen=True)
class Suggestion:
    """A fix for ``text[start:end]``, options best first."""

    id: str
    start: int
    end: int
    original: str
    kind: str
    explanation: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class DocumentState:
    """The service's view of one document version."""

    version: int
    state: str
    suggestions: tuple[Suggestion, ...]

    @property
    def settled(self) -> bool:
        """Whether no check of this version is pending or running."""
        return self.state in {"done", "failed"}


def parse_version(data: object) -> int:
    """The ``version`` of a ``PUT /documents/{uri}`` reply."""
    return _int(_mapping(data, "reply"), "version")


def parse_document(data: object) -> DocumentState:
    """A ``GET /documents/{uri}/suggestions`` reply."""
    body = _mapping(data, "reply")
    suggestions = body.get("suggestions")
    if not isinstance(suggestions, list):
        msg = "suggestions must be a list"
        raise SchemaError(msg)
    return DocumentState(
        version=_int(body, "version"),
        state=_str(body, "state"),
        suggestions=tuple(_suggestion(item) for item in suggestions),
    )


def _suggestion(data: object) -> Suggestion:
    body = _mapping(data, "suggestion")
    options = body.get("options")
    if not isinstance(options, list):
        msg = "options must be a list"
        raise SchemaError(msg)
    start, end = _int(body, "start"), _int(body, "end")
    if not 0 <= start <= end:
        msg = f"bad span [{start}, {end})"
        raise SchemaError(msg)
    return Suggestion(
        id=_str(body, "id"),
        start=start,
        end=end,
        original=_str(body, "original"),
        kind=_str(body, "kind"),
        explanation=_str(body, "explanation"),
        options=tuple(_str(_mapping(option, "option"), "replacement") for option in options),
    )


def _mapping(data: object, what: str) -> dict[str, object]:
    if not isinstance(data, dict):
        msg = f"{what} must be an object"
        raise SchemaError(msg)
    return {str(key): value for key, value in data.items()}


def _int(body: dict[str, object], key: str) -> int:
    value = body.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise SchemaError(msg)
    return value


def _str(body: dict[str, object], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise SchemaError(msg)
    return value
