"""Interfaces for chat-completion LLMs."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.llm.models import ChatMessage, ChatOptions


class LLMClient(Protocol):
    """Answers a chat with one reply.

    Implementations are safe to call from several threads at once and limit how many
    requests run in parallel themselves.
    """

    def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """The model's reply text to ``messages``.

        Raises ``LLMError`` when the server gives no usable reply after retries.
        """
        ...


class AsyncLLMClient(Protocol):
    """Answers a chat with one reply without blocking the event loop.

    Implementations limit how many requests run in parallel themselves.
    """

    async def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """The model's reply text to ``messages``.

        Raises ``LLMError`` when the server gives no usable reply after retries.
        """
        ...
