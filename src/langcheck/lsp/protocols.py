"""Protocols of the LSP proxy."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from langcheck.api.schemas import DocumentBody


class DocumentsService(Protocol):
    """The parts of the langcheck service the proxy uses (``ServiceClient`` has them)."""

    async def put_document(self, uri: str, text: str) -> int:
        """Open or update the document; the server's new version."""
        ...

    async def close_document(self, uri: str) -> bool:
        """Close the document."""
        ...

    async def dismiss(self, uri: str, suggestion_id: str) -> bool:
        """Dismiss a suggestion."""
        ...

    async def add_words(self, uri: str, words: Sequence[str]) -> list[str]:
        """Add words to the document's dictionary."""
        ...

    def events(self, uri: str) -> AsyncIterator[DocumentBody | None]:
        """Snapshots of the document; ``None`` when it closed."""
        ...
