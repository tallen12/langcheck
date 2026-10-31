"""Building a :class:`DocumentContext` with one LLM call over the document."""

import json
from collections.abc import Collection, Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from langcheck.context.models import DocumentContext, Register
from langcheck.llm.models import ChatMessage, ChatOptions, Role
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.llm.structured import Failed, Parsed, complete_structured
from langcheck.prompts.templates import load_template
from langcheck.text.segment import paragraph_spans

CONTEXT_PROMPT_VERSION = "context-v1"
DEFAULT_MAX_PARAGRAPHS = 30
GAP = "[...]"


class ContextReply(BaseModel):
    """The LLM's description of a document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    # Aliased because ``register`` would shadow ``BaseModel.register``.
    writing_register: Annotated[Register, Field(alias="register")]
    dialect: str
    glossary: Annotated[list[str], Field(max_length=200)]
    notes: str


class NoContext:
    """A ``ContextSource`` for a checker that reads no context: it costs no LLM call."""

    async def build(
        self, text: str, *, dictionary: Collection[str] = ()
    ) -> DocumentContext | Failed:
        """The empty context."""
        return DocumentContext.empty()


class ContextBuilder:
    """Asks the LLM to describe a document, sampling long documents.

    A document with more than ``max_paragraphs`` paragraphs is shown as its first and
    last thirds of that budget plus evenly spaced paragraphs from the middle.
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        *,
        options: ChatOptions,
        max_paragraphs: int = DEFAULT_MAX_PARAGRAPHS,
    ) -> None:
        if max_paragraphs < 3:
            msg = f"max_paragraphs must be at least 3, got {max_paragraphs}"
            raise ValueError(msg)
        self._llm = llm
        self._options = options
        self._max_paragraphs = max_paragraphs

    async def build(
        self, text: str, *, dictionary: Collection[str] = ()
    ) -> DocumentContext | Failed:
        """The context of ``text``; ``dictionary`` words always join the glossary.

        Returns ``Failed`` when the LLM gives no valid reply; callers fall back to
        :meth:`DocumentContext.empty`.
        """
        messages = context_messages(sample(text, max_paragraphs=self._max_paragraphs))
        match await complete_structured(self._llm, messages, ContextReply, options=self._options):
            case Parsed(value=reply):
                terms = (t.strip() for t in [*reply.glossary, *sorted(dictionary)])
                glossary = dict.fromkeys(terms)
                return DocumentContext(
                    summary=reply.summary.strip(),
                    register=reply.writing_register,
                    dialect=reply.dialect.strip(),
                    glossary=tuple(term for term in glossary if term),
                    notes=reply.notes.strip(),
                )
            case Failed() as failed:
                return failed


def sample(text: str, *, max_paragraphs: int) -> str:
    """``text``, or for a long one its start, end and a spread of middle paragraphs."""
    paragraphs = [text[s.start : s.end] for s in paragraph_spans(text)]
    if len(paragraphs) <= max_paragraphs:
        return "\n\n".join(paragraphs)
    edge = max_paragraphs // 3
    middle_count = max_paragraphs - 2 * edge
    middle = paragraphs[edge:-edge]
    step = len(middle) / middle_count
    picked = [middle[int(i * step)] for i in range(middle_count)]
    parts: Sequence[str] = [
        "\n\n".join(paragraphs[:edge]),
        "\n\n".join(picked),
        "\n\n".join(paragraphs[-edge:]),
    ]
    return f"\n\n{GAP}\n\n".join(parts)


def context_messages(document: str) -> list[ChatMessage]:
    """The chat that asks for the context of ``document``."""
    system = load_template(CONTEXT_PROMPT_VERSION).substitute(
        schema=json.dumps(ContextReply.model_json_schema(), indent=2)
    )
    return [ChatMessage(Role.SYSTEM, system), ChatMessage(Role.USER, document)]
