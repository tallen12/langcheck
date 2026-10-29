"""``DictionaryStore`` in memory, for tests, throwaway servers and unsaved documents."""

from collections.abc import Collection

from langcheck.dictionary.words import clean_words


class MemoryDictionaryStore:
    """Keeps words in a dictionary keyed by (user, URI); lost on restart."""

    def __init__(self) -> None:
        self._words: dict[tuple[str, str], frozenset[str]] = {}

    def words(self, user: str, uri: str) -> frozenset[str]:
        """The document's words."""
        return self._words.get((user, uri), frozenset[str]())

    def add(self, user: str, uri: str, words: Collection[str]) -> None:
        """Add ``words``."""
        self._words[(user, uri)] = self.words(user, uri) | clean_words(words)

    def remove(self, user: str, uri: str, words: Collection[str]) -> None:
        """Remove ``words``."""
        remaining = self.words(user, uri) - {w.strip() for w in words}
        if remaining:
            self._words[(user, uri)] = remaining
        else:
            self._words.pop((user, uri), None)

    def rename(self, user: str, old_uri: str, new_uri: str) -> None:
        """Move the words to ``new_uri``."""
        moved = self._words.pop((user, old_uri), frozenset[str]())
        if moved:
            self._words[(user, new_uri)] = self.words(user, new_uri) | moved
