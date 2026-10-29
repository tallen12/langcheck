import asyncio

from pydantic import BaseModel

from langcheck.llm.models import ChatMessage, ChatOptions, LLMError, Role
from langcheck.llm.structured import Failed, Parsed, complete_structured, strip_fence
from tests.conftest import ScriptedLLM

CHAT = [ChatMessage(Role.USER, "Count.")]


class Count(BaseModel):
    n: int


def ask(llm: ScriptedLLM) -> Parsed[Count] | Failed:
    return asyncio.run(complete_structured(llm, CHAT, Count, options=ChatOptions()))


def test_parses_a_valid_reply_and_sends_the_schema() -> None:
    llm = ScriptedLLM(['{"n": 3}'])

    assert ask(llm) == Parsed(Count(n=3))
    assert llm.options[0].json_schema == Count.model_json_schema()
    assert len(llm.chats) == 1


def test_retries_once_with_the_validation_error() -> None:
    llm = ScriptedLLM(['{"n": "many"}', '{"n": 4}'])

    assert ask(llm) == Parsed(Count(n=4))
    retry = llm.chats[1]
    assert retry[:2] == [CHAT[0], ChatMessage(Role.ASSISTANT, '{"n": "many"}')]
    assert retry[2].role is Role.USER
    assert "did not match the schema" in retry[2].content


def test_fails_after_two_invalid_replies() -> None:
    llm = ScriptedLLM(["not json"])

    result = ask(llm)

    assert isinstance(result, Failed)
    assert result.reason.startswith("invalid reply")
    assert len(llm.chats) == 2


def test_reports_llm_errors_without_retrying() -> None:
    llm = ScriptedLLM([LLMError("server down")])

    assert ask(llm) == Failed("LLM error: server down")
    assert len(llm.chats) == 1


def test_accepts_replies_wrapped_in_a_code_fence() -> None:
    assert ask(ScriptedLLM(['```json\n{"n": 1}\n```'])) == Parsed(Count(n=1))
    assert strip_fence('```\n{"n": 1}\n```') == '{"n": 1}'
    assert strip_fence('{"n": 1}') == '{"n": 1}'
