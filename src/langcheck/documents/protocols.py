"""Protocols of the documents domain."""

from typing import Protocol

from langcheck.documents.models import Document, DocumentKey


class DocumentStore(Protocol):
    """Holds the current version of every open document.

    Documents are sessions: they may be lost on restart, and clients send the text again
    when they reconnect.
    """

    async def get(self, key: DocumentKey) -> Document | None:
        """The document at ``key``, or ``None`` when it is not open."""
        ...

    async def put(self, document: Document) -> None:
        """Store ``document`` under its key, replacing any earlier version."""
        ...

    async def delete(self, key: DocumentKey) -> bool:
        """Close the document at ``key``; ``False`` when it was not open."""
        ...
