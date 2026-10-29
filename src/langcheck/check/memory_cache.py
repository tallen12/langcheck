"""An in-memory :class:`ResultCache` with least-recently-used eviction."""

from collections import OrderedDict

from langcheck.check.cache_models import CacheKey, SentenceResult

DEFAULT_MAX_ENTRIES = 100_000


class MemoryResultCache:
    """Keeps at most ``max_entries`` results in this process, evicting the least recently
    used. Lost on restart.
    """

    def __init__(self, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            msg = f"max_entries must be positive, got {max_entries}"
            raise ValueError(msg)
        self._max_entries = max_entries
        self._entries: OrderedDict[CacheKey, SentenceResult] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    async def get(self, key: CacheKey) -> SentenceResult | None:
        """The stored result for ``key``, marking it as recently used."""
        result = self._entries.get(key)
        if result is not None:
            self._entries.move_to_end(key)
        return result

    async def put(self, key: CacheKey, result: SentenceResult) -> None:
        """Store ``result``, evicting the least recently used entry when full."""
        self._entries[key] = result
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
