"""Keeping unsaved documents' words out of persistent storage."""

from collections.abc import Collection

from langcheck.dictionary.protocols import DictionaryStore
from langcheck.documents.models import UNTITLED_SCHEME


class UntitledInMemory:
    """Sends ``untitled:`` documents to ``scratch`` and the rest to ``persistent``, so
    nothing about unsaved documents is stored. Renaming an unsaved document to a real
    URI (save as) moves its words into persistent storage.
    """

    def __init__(self, persistent: DictionaryStore, *, scratch: DictionaryStore) -> None:
        self._persistent = persistent
        self._scratch = scratch

    def _store(self, uri: str) -> DictionaryStore:
        return self._scratch if uri.startswith(UNTITLED_SCHEME) else self._persistent

    def words(self, user: str, uri: str) -> frozenset[str]:
        """The document's words."""
        return self._store(uri).words(user, uri)

    def add(self, user: str, uri: str, words: Collection[str]) -> None:
        """Add ``words``."""
        self._store(uri).add(user, uri, words)

    def remove(self, user: str, uri: str, words: Collection[str]) -> None:
        """Remove ``words``."""
        self._store(uri).remove(user, uri, words)

    def rename(self, user: str, old_uri: str, new_uri: str) -> None:
        """Move the words, across stores when the document is saved or unsaved."""
        source, target = self._store(old_uri), self._store(new_uri)
        if source is target:
            source.rename(user, old_uri, new_uri)
            return
        moved = source.words(user, old_uri)
        source.remove(user, old_uri, moved)
        target.add(user, new_uri, moved)
