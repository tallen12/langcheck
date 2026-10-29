import asyncio
from collections.abc import Collection

from langcheck.check.cache_models import CacheKey, SentenceResult
from langcheck.check.checker import CheckResult
from langcheck.check.protocols import ResultCache, TextChecker
from langcheck.context.models import DocumentContext


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


class SilentChecker:
    """Finds nothing."""

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        return CheckResult((), (), ())


def test_text_checker_contract() -> None:
    checker: TextChecker = SilentChecker()

    result = asyncio.run(checker.check("Hi.", user="ana", context=DocumentContext.empty()))

    assert result == CheckResult((), (), ())
