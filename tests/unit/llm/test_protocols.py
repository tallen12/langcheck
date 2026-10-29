import asyncio
from collections.abc import Sequence

from langcheck.llm.models import ChatMessage, ChatOptions, Role
from langcheck.llm.protocols import AsyncLLMClient, LLMClient


class EchoClient:
    """Replies with the last message's text."""

    def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        return messages[-1].content


def test_client_contract() -> None:
    client: LLMClient = EchoClient()

    assert client.complete([ChatMessage(Role.USER, "hi")], options=ChatOptions()) == "hi"


class AsyncEchoClient:
    """Replies with the last message's text, asynchronously."""

    async def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        return messages[-1].content


def test_async_client_contract() -> None:
    client: AsyncLLMClient = AsyncEchoClient()

    reply = asyncio.run(client.complete([ChatMessage(Role.USER, "hi")], options=ChatOptions()))

    assert reply == "hi"
