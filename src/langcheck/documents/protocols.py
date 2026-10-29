"""Protocols of the documents domain."""

import asyncio
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol

from langcheck.documents.events import DocumentEvent
from langcheck.documents.models import Document, DocumentKey

if TYPE_CHECKING:
    # Only for annotations: sessions.py imports this module for DocumentStore.
    from langcheck.documents.sessions import AcceptResult, NotOpen, UpdateResult


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


class DocumentService(Protocol):
    """Open documents for clients: updates, results, accept, dismiss and events."""

    async def get(self, key: DocumentKey) -> Document | None:
        """The current version of the document."""
        ...

    async def update(
        self, key: DocumentKey, text: str, *, base_version: int | None = None
    ) -> UpdateResult:
        """Open or update the document and schedule its check."""
        ...

    async def accept(
        self, key: DocumentKey, suggestion_id: str, *, option: int, base_version: int | None
    ) -> AcceptResult:
        """Apply an option of a current suggestion as a new version."""
        ...

    async def dismiss(self, key: DocumentKey, suggestion_id: str) -> bool:
        """Hide a suggestion for the rest of the session."""
        ...

    async def dictionary_changed(self, key: DocumentKey) -> bool:
        """Re-check the open document after its dictionary changed."""
        ...

    async def close(self, key: DocumentKey) -> bool:
        """Close the document."""
        ...

    async def rename(self, key: DocumentKey, new_uri: str) -> UpdateResult | NotOpen:
        """Move the document to a new URI."""
        ...

    def watch(self, key: DocumentKey) -> AbstractAsyncContextManager[asyncio.Queue[DocumentEvent]]:
        """Events for the document, starting with the current snapshot."""
        ...
