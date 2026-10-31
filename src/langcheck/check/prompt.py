"""The chat sent to the LLM for one chunk (prompt version ``check-v1``)."""

import json

from langcheck.check.chunk import Chunk
from langcheck.check.reply import CheckReply
from langcheck.context.models import DocumentContext
from langcheck.llm.models import ChatMessage, Role
from langcheck.prompts.templates import load_template
from langcheck.taxonomy.kinds import ErrorKind

PROMPT_VERSION = "check-v2"


def system_prompt() -> str:
    """The fixed instructions, identical for every chunk so servers can cache them.

    They include the JSON Schema of :class:`CheckReply`, which is also sent as
    ``json_schema`` for servers that honour it.
    """
    return load_template(PROMPT_VERSION).substitute(
        kinds="\n".join(f"- {kind.value}" for kind in ErrorKind),
        schema=json.dumps(CheckReply.model_json_schema(), indent=2),
    )


def check_messages(chunk: Chunk, context: DocumentContext) -> list[ChatMessage]:
    """The system and user messages that ask the model to check ``chunk``."""
    return [
        ChatMessage(Role.SYSTEM, system_prompt()),
        ChatMessage(Role.USER, user_prompt(chunk, context)),
    ]


def user_prompt(chunk: Chunk, context: DocumentContext) -> str:
    """The document context, the read-only neighbours and the numbered focus sentences."""
    sections = [
        "Document context:",
        f"- Summary: {context.summary or 'unknown'}",
        f"- Register: {context.register.value}",
        f"- Dialect: {context.dialect or 'unknown'}",
        f"- Notes: {context.notes or 'none'}",
    ]
    if chunk.before:
        sections += ["", "Context paragraph before (do not check):", chunk.before]
    sections += ["", "Sentences to check:"]
    sections += [f"[{s.id}] {s.text}" for s in chunk.sentences]
    if chunk.after:
        sections += ["", "Context paragraph after (do not check):", chunk.after]
    return "\n".join(sections)
