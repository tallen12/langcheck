import pytest

from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.dictionary.protocols import DictionaryStore
from langcheck.dictionary.words import InvalidWordError

A = "file:///a.md"
B = "file:///b.md"


def check_contract(store: DictionaryStore) -> None:
    """The behavior every DictionaryStore shares."""
    store.add("ana", A, [" Bilbo ", "Shire", "Shire"])
    store.add("ana", A, ["Bilbo"])
    store.add("ben", A, ["Gandalf"])

    assert store.words("ana", A) == {"Bilbo", "Shire"}
    assert store.words("ben", A) == {"Gandalf"}
    assert store.words("ana", B) == frozenset()

    store.remove("ana", A, ["Shire", "unknown"])
    assert store.words("ana", A) == {"Bilbo"}

    store.add("ana", B, ["Frodo"])
    store.rename("ana", A, B)
    assert store.words("ana", A) == frozenset()
    assert store.words("ana", B) == {"Bilbo", "Frodo"}
    store.rename("ana", "file:///none.md", A)
    assert store.words("ana", A) == frozenset()

    store.remove("ana", B, ["Bilbo", "Frodo"])
    assert store.words("ana", B) == frozenset()
    with pytest.raises(InvalidWordError):
        store.add("ana", A, ["  "])


def test_memory_store_contract() -> None:
    check_contract(MemoryDictionaryStore())
