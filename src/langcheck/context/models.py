"""The document context sent with every chunk: what the text is and what not to flag."""

import hashlib
import json
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
    spellings that must not be flagged.
    """

    summary: str
    register: Register
    dialect: str
    glossary: tuple[str, ...]
    notes: str

    @property
    def version(self) -> str:
        """A fingerprint of the content, part of every result cache key.

        Equal contexts share it, so results made with a different context are never
        reused, while a rebuilt but unchanged context keeps its cache.
        """
        fields = [self.summary, self.register.value, self.dialect, self.glossary, self.notes]
        encoded = json.dumps(fields, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @classmethod
    def empty(cls) -> DocumentContext:
        """A context that says nothing, for checks without a built context."""
        return cls(
            summary="",
            register=Register.NEUTRAL,
            dialect="",
            glossary=(),
            notes="",
        )
