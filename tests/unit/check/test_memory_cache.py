import asyncio

import pytest

from langcheck.check.cache_models import CacheKey, SentenceResult
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.protocols import ResultCache

EMPTY = SentenceResult(())


def key(name: str) -> CacheKey:
    return CacheKey("ana", name)


def test_stores_and_returns_results() -> None:
    cache: ResultCache = MemoryResultCache()

    async def run() -> tuple[SentenceResult | None, SentenceResult | None]:
        await cache.put(key("a"), EMPTY)
        return await cache.get(key("a")), await cache.get(key("b"))

    assert asyncio.run(run()) == (EMPTY, None)


def test_evicts_the_least_recently_used() -> None:
    cache = MemoryResultCache(max_entries=2)

    async def run() -> list[SentenceResult | None]:
        await cache.put(key("a"), EMPTY)
        await cache.put(key("b"), EMPTY)
        await cache.get(key("a"))
        await cache.put(key("c"), EMPTY)
        return [await cache.get(key(name)) for name in "abc"]

    assert asyncio.run(run()) == [EMPTY, None, EMPTY]
    assert len(cache) == 2


def test_needs_room_for_one_entry() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        MemoryResultCache(max_entries=0)
