"""Protocols of the check domain."""

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol

from langcheck.check.cache_models import CacheKey, SentenceResult
from langcheck.context.models import DocumentContext

if TYPE_CHECKING:
    # Only for annotations: checker.py imports this module for ResultCache.
    from langcheck.check.checker import CheckResult


class ResultCache(Protocol):
    """Stores each sentence's suggestions so an unchanged sentence is never sent again.

    Keys are scoped by user (see :func:`langcheck.check.cache_models.cache_key`); the
    cache is never shared across users.
    """

    async def get(self, key: CacheKey) -> SentenceResult | None:
        """The stored result for ``key``, or ``None`` on a miss."""
        ...

    async def put(self, key: CacheKey, result: SentenceResult) -> None:
        """Store ``result`` under ``key``, possibly evicting older entries."""
        ...


class TextChecker(Protocol):
    """Checks a whole text for one user."""

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        """The suggestions for ``text``; ``dictionary`` words are never flagged."""
        ...
