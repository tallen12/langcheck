import asyncio
import json

import pytest

from langcheck.context.builder import (
    GAP,
    ContextBuilder,
    ContextReply,
    context_messages,
    sample,
)
from langcheck.context.models import DocumentContext, Register
from langcheck.llm.models import ChatOptions, LLMError, Role
from langcheck.llm.structured import Failed
from tests.conftest import ScriptedLLM

REPLY = json.dumps(
    {
        "summary": " A hobbit story. ",
        "register": "fiction-dialogue-heavy",
        "dialect": "en-GB",
        "glossary": ["Bilbo", "Shire", " "],
        "notes": "Past tense.",
    }
)


def build(llm: ScriptedLLM, text: str = "Bilbo left.") -> DocumentContext | Failed:
    builder = ContextBuilder(llm, options=ChatOptions(), max_paragraphs=3)
    return asyncio.run(builder.build(text, dictionary=["Gandalf", "Bilbo"]))


def test_builds_the_context_and_adds_dictionary_words() -> None:
    llm = ScriptedLLM([REPLY])

    context = build(llm)

    assert context == DocumentContext(
        summary="A hobbit story.",
        register=Register.FICTION_DIALOGUE,
        dialect="en-GB",
        glossary=("Bilbo", "Shire", "Gandalf"),
        notes="Past tense.",
    )
    assert llm.chats[0][1].content == "Bilbo left."
    assert llm.options[0].json_schema == ContextReply.model_json_schema()


def test_reports_failure_when_the_llm_gives_no_valid_reply() -> None:
    assert isinstance(build(ScriptedLLM(['{"summary": "no"}'])), Failed)
    assert isinstance(build(ScriptedLLM([LLMError("down")])), Failed)


def test_needs_room_for_start_middle_and_end() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        ContextBuilder(ScriptedLLM([REPLY]), options=ChatOptions(), max_paragraphs=2)


def test_sample_keeps_short_documents_whole() -> None:
    assert sample("One.\n\n\nTwo.", max_paragraphs=3) == "One.\n\nTwo."


def test_sample_takes_start_middle_and_end_of_long_documents() -> None:
    text = "\n\n".join(f"P{i}." for i in range(20))

    sampled = sample(text, max_paragraphs=6)

    assert sampled.split(f"\n\n{GAP}\n\n") == [
        "P0.\n\nP1.",
        "P2.\n\nP10.",
        "P18.\n\nP19.",
    ]


def test_context_messages_hold_the_schema_and_the_document() -> None:
    system, user = context_messages("Doc.")

    assert system.role is Role.SYSTEM
    assert json.dumps(ContextReply.model_json_schema(), indent=2) in system.content
    assert "$schema" not in system.content
    assert (user.role, user.content) == (Role.USER, "Doc.")
