import asyncio
from collections.abc import Callable, Sequence

import pytest

from langcheck.auth.models import User, UserExistsError
from langcheck.auth.protocols import ApiKeyStore
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError, Role
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import AsyncSentenceScorer, SentenceScorer
from langcheck.text.models import Span
from tests.conftest import (
    MemoryApiKeyStore,
    Responder,
    ScriptedLLM,
    SyncTableScorer,
    TableScorer,
)

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


def test_memory_api_key_store_keeps_one_key_per_user() -> None:
    store: ApiKeyStore = MemoryApiKeyStore()

    store.add("ana", "h1")
    with pytest.raises(UserExistsError):
        store.add("ana", "h2")

    assert store.user_for("h1") == User("ana")
    assert store.user_for("h2") is None
    assert store.users() == ["ana"]
    assert store.revoke("ana") is True
    assert store.revoke("ana") is False


def test_table_scorers_look_scores_up_and_record_batches() -> None:
    texts = [ScoredText("a", Span(0, 1)), ScoredText("b", Span(0, 1))]
    asynchronous = TableScorer({"a": 1.5}, default=-1.0)
    blocking = SyncTableScorer({"b": 2.0})
    async_contract: AsyncSentenceScorer = asynchronous
    sync_contract: SentenceScorer = blocking

    assert asyncio.run(async_contract.score(texts)) == [1.5, -1.0]
    assert sync_contract.score(texts) == [0.0, 2.0]
    assert asynchronous.batches == [texts]
    assert blocking.batches == [texts]
