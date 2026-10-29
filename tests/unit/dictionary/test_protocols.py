from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.dictionary.protocols import DictionaryStore


def test_memory_store_satisfies_the_protocol() -> None:
    store: DictionaryStore = MemoryDictionaryStore()

    assert store.words("ana", "file:///a.md") == frozenset()
