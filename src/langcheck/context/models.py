"""The document context sent with every chunk: what the text is and what not to flag."""

from dataclasses import dataclass
from enum import StrEnum


class Register(StrEnum):
    """How formal the writing is."""

    FORMAL = "formal"
    NEUTRAL = "neutral"
    INFORMAL = "informal"
    FICTION_DIALOGUE = "fiction-dialogue-heavy"


@dataclass(frozen=True, slots=True)
class DocumentContext:
    """A short summary of a whole document, built once and sent with every chunk.

    ``glossary`` holds names, product terms, invented words and intentional nonstandard
    spellings that must not be flagged. ``version`` changes whenever the context is
    rebuilt, so cached results made with an older context are not reused.
    """

    summary: str
    register: Register
    dialect: str
    glossary: tuple[str, ...]
    notes: str
    version: int

    @classmethod
    def empty(cls) -> DocumentContext:
        """A context that says nothing, for checks without a built context."""
        return cls(
            summary="",
            register=Register.NEUTRAL,
            dialect="",
            glossary=(),
            notes="",
            version=0,
        )
