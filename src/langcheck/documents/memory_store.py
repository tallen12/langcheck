"""``DocumentStore`` in this process's memory."""

from langcheck.documents.models import Document, DocumentKey


class MemoryDocumentStore:
    """Keeps open documents in a dictionary; lost on restart."""

    def __init__(self) -> None:
        self._documents: dict[DocumentKey, Document] = {}

    def __len__(self) -> int:
        return len(self._documents)

    async def get(self, key: DocumentKey) -> Document | None:
        """The document at ``key``."""
        return self._documents.get(key)

    async def put(self, document: Document) -> None:
        """Store ``document``."""
        self._documents[document.key] = document

    async def delete(self, key: DocumentKey) -> bool:
        """Close the document at ``key``."""
        return self._documents.pop(key, None) is not None
