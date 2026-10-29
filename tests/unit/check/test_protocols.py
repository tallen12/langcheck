import asyncio

from langcheck.check.cache_models import CacheKey, SentenceResult
from langcheck.check.protocols import ResultCache


class NullCache:
    """A cache that never stores anything."""

    async def get(self, key: CacheKey) -> SentenceResult | None:
        return None

    async def put(self, key: CacheKey, result: SentenceResult) -> None:
        return None


def test_result_cache_contract() -> None:
    cache: ResultCache = NullCache()

    async def run() -> SentenceResult | None:
        await cache.put(CacheKey("ana", "k"), SentenceResult(()))
        return await cache.get(CacheKey("ana", "k"))

    assert asyncio.run(run()) is None
