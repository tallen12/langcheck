"""Fixtures shared across test types."""

from collections.abc import Callable, Sequence

import pytest

from langcheck.auth.models import User, UserExistsError
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError
from langcheck.scoring.models import ScoredText

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


class MemoryApiKeyStore:
    """An ``ApiKeyStore`` fake that keeps key hashes in a dictionary."""

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def user_for(self, token_hash: str) -> User | None:
        """The user whose key has ``token_hash``."""
        return next((User(n) for n, h in self.hashes.items() if h == token_hash), None)

    def add(self, name: str, token_hash: str) -> None:
        """Store the hash; refuse a second key for one user."""
        if name in self.hashes:
            raise UserExistsError(name)
        self.hashes[name] = token_hash

    def revoke(self, name: str) -> bool:
        """Remove ``name``'s key."""
        return self.hashes.pop(name, None) is not None

    def users(self) -> list[str]:
        """Every user with a key, sorted."""
        return sorted(self.hashes)


class TableScorer:
    """A ``SentenceScorer`` and ``AsyncSentenceScorer`` fake that looks scores up by text.

    Unknown texts score ``default``. Every batch it is asked to score is recorded.
    """

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        self._table = table
        self._default = default
        self.batches: list[list[ScoredText]] = []

    def score_now(self, texts: Sequence[ScoredText]) -> list[float]:
        """Score ``texts`` and record the batch."""
        self.batches.append(list(texts))
        return [self._table.get(t.text, self._default) for t in texts]

    async def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """The async form, for rankers."""
        return self.score_now(texts)


class SyncTableScorer:
    """The blocking form of :class:`TableScorer`, for code that runs scorers in threads."""

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        self.table = TableScorer(table, default=default)

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """Score ``texts``."""
        return self.table.score_now(texts)

    @property
    def batches(self) -> list[list[ScoredText]]:
        """Every batch scored so far."""
        return self.table.batches
