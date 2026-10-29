from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.protocols import DocumentStore


def test_memory_store_satisfies_the_protocol() -> None:
    store: DocumentStore = MemoryDocumentStore()

    assert store is not None
