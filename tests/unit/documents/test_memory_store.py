import asyncio

from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.models import CheckState, Document, DocumentKey
from langcheck.documents.protocols import DocumentStore

KEY = DocumentKey("ana", "file:///a.md")


def test_puts_gets_replaces_and_deletes() -> None:
    store = MemoryDocumentStore()
    contract: DocumentStore = store
    first = Document(key=KEY, text="a", version=1, state=CheckState.PENDING)
    second = Document(key=KEY, text="b", version=2, state=CheckState.PENDING)

    async def run() -> tuple[Document | None, Document | None, bool, bool]:
        await contract.put(first)
        await contract.put(second)
        found = await contract.get(KEY)
        other = await contract.get(DocumentKey("ben", "file:///a.md"))
        return found, other, await contract.delete(KEY), await contract.delete(KEY)

    assert asyncio.run(run()) == (second, None, True, False)
    assert len(store) == 0
