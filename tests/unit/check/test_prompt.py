import json

from langcheck.check.chunk import Chunk
from langcheck.check.prompt import (
    PROMPT_VERSION,
    check_messages,
    system_prompt,
    user_prompt,
)
from langcheck.check.reply import CheckReply
from langcheck.context.models import DocumentContext, Register
from langcheck.llm.models import Role
from langcheck.taxonomy.kinds import ErrorKind
from langcheck.text.models import Sentence, Span

CHUNK = Chunk(
    sentences=(
        Sentence("S2", Span(10, 22), "Bilbo runned."),
        Sentence("S3", Span(23, 30), "Its late."),
    ),
    before="Before text.",
    after="After text.",
)
CONTEXT = DocumentContext(
    summary="A hobbit story.",
    register=Register.FICTION_DIALOGUE,
    dialect="en-GB",
    glossary=("Bilbo",),
    notes="Past tense, third person.",
    version=2,
)


def test_system_prompt_lists_every_kind_and_the_schema() -> None:
    prompt = system_prompt()

    assert PROMPT_VERSION == "check-v1"
    assert "$" not in prompt.replace("$defs", "").replace("$ref", "")
    for kind in ErrorKind:
        assert f"- {kind.value}\n" in prompt
    assert json.dumps(CheckReply.model_json_schema(), indent=2) in prompt


def test_user_prompt_has_context_neighbours_and_numbered_sentences() -> None:
    prompt = user_prompt(CHUNK, CONTEXT, dictionary=["Shire", "Bilbo"])

    assert "- Summary: A hobbit story." in prompt
    assert "- Register: fiction-dialogue-heavy" in prompt
    assert "- Dialect: en-GB" in prompt
    assert "- Glossary (never flag): Bilbo, Shire" in prompt
    assert "- Notes: Past tense, third person." in prompt
    assert prompt.index("Before text.") < prompt.index("[S2] Bilbo runned.")
    assert prompt.index("[S2] Bilbo runned.") < prompt.index("[S3] Its late.")
    assert prompt.index("[S3] Its late.") < prompt.index("After text.")


def test_user_prompt_leaves_out_missing_context() -> None:
    chunk = Chunk(sentences=CHUNK.sentences, before="", after="")

    prompt = user_prompt(chunk, DocumentContext.empty())

    assert "do not check" not in prompt
    assert "- Summary: unknown" in prompt
    assert "- Glossary (never flag): none" in prompt


def test_check_messages_are_system_then_user() -> None:
    messages = check_messages(CHUNK, CONTEXT)

    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert messages[0].content == system_prompt()
    assert messages[1].content == user_prompt(CHUNK, CONTEXT)
