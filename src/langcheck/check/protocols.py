"""Protocols of the check domain."""

from typing import Protocol

from langcheck.check.cache_models import CacheKey, SentenceResult


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
