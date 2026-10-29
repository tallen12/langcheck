import asyncio
from collections.abc import Callable, Sequence

import pytest

from langcheck.llm.models import ChatMessage, ChatOptions, LLMError, Role
from langcheck.llm.protocols import AsyncLLMClient
from tests.conftest import Responder, ScriptedLLM

HI = [ChatMessage(Role.USER, "hi")]


def test_scripted_llm_replies_in_order_repeats_the_last_and_records_chats(
    scripted_llm: Callable[[Sequence[str | LLMError | Responder]], ScriptedLLM],
) -> None:
    llm = scripted_llm(["one", lambda chat: chat[-1].content.upper()])
    client: AsyncLLMClient = llm

    async def run() -> list[str]:
        return [await client.complete(HI, options=ChatOptions()) for _ in range(3)]

    assert asyncio.run(run()) == ["one", "HI", "HI"]
    assert llm.chats == [HI, HI, HI]
    assert len(llm.options) == 3


def test_scripted_llm_raises_scripted_errors() -> None:
    llm = ScriptedLLM([LLMError("down")])

    with pytest.raises(LLMError, match="down"):
        asyncio.run(llm.complete(HI, options=ChatOptions()))


def test_scripted_llm_needs_a_script() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        ScriptedLLM([])
