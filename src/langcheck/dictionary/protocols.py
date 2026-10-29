"""Protocols of the dictionary domain."""

from collections.abc import Collection
from typing import Protocol


class DictionaryStore(Protocol):
    """The words each user has added to each document (by URI). Calls may block (SQL);
    async callers run them in a worker thread.
    """

    def words(self, user: str, uri: str) -> frozenset[str]:
        """The document's words; empty for an unknown document."""
        ...

    def add(self, user: str, uri: str, words: Collection[str]) -> None:
        """Add ``words``; words already there are ignored."""
        ...

    def remove(self, user: str, uri: str, words: Collection[str]) -> None:
        """Remove ``words``; words not there are ignored."""
        ...

    def rename(self, user: str, old_uri: str, new_uri: str) -> None:
        """Move the document's words to ``new_uri``, merging with any already there."""
        ...
