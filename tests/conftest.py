"""Fixtures shared across test types."""

from collections.abc import Callable, Sequence

import pytest

from langcheck.llm.models import ChatMessage, ChatOptions, LLMError

type Responder = Callable[[Sequence[ChatMessage]], str | LLMError]


class ScriptedLLM:
    """An ``AsyncLLMClient`` fake that answers from a script and records every chat.

    Each entry is a reply, an ``LLMError`` to raise, or a function of the chat returning
    either, for replies that depend on the request (such as which sentences it holds).
    The last entry repeats once the script runs out.
    """

    def __init__(self, script: Sequence[str | LLMError | Responder]) -> None:
        if not script:
            msg = "the script needs at least one entry"
            raise ValueError(msg)
        self._script = list(script)
        self.chats: list[list[ChatMessage]] = []
        self.options: list[ChatOptions] = []

    async def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """Return the next scripted reply for ``messages``."""
        self.chats.append(list(messages))
        self.options.append(options)
        entry = self._script[min(len(self.chats), len(self._script)) - 1]
        reply = entry if isinstance(entry, (str, LLMError)) else entry(messages)
        match reply:
            case LLMError():
                raise reply
            case str():
                return reply


@pytest.fixture
def scripted_llm() -> Callable[[Sequence[str | LLMError | Responder]], ScriptedLLM]:
    """Build a :class:`ScriptedLLM` from a script."""
    return ScriptedLLM
