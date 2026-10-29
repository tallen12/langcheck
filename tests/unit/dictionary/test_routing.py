from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.dictionary.routing import UntitledInMemory
from tests.unit.dictionary.test_memory_store import check_contract

UNTITLED = "untitled:5e1f"


def test_routing_store_contract() -> None:
    check_contract(UntitledInMemory(MemoryDictionaryStore(), scratch=MemoryDictionaryStore()))


def test_unsaved_documents_never_reach_persistent_storage_until_saved() -> None:
    persistent = MemoryDictionaryStore()
    scratch = MemoryDictionaryStore()
    store = UntitledInMemory(persistent, scratch=scratch)

    store.add("ana", UNTITLED, ["Bilbo"])
    assert persistent.words("ana", UNTITLED) == frozenset()
    assert store.words("ana", UNTITLED) == {"Bilbo"}

    store.rename("ana", UNTITLED, "file:///saved.md")
    assert scratch.words("ana", UNTITLED) == frozenset()
    assert persistent.words("ana", "file:///saved.md") == {"Bilbo"}

    store.rename("ana", "file:///saved.md", "untitled:again")
    assert persistent.words("ana", "file:///saved.md") == frozenset()
    assert scratch.words("ana", "untitled:again") == {"Bilbo"}
