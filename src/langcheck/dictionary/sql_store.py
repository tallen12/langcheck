"""``DictionaryStore`` in the SQL database."""

from collections.abc import Collection

from sqlalchemy import Column, Engine, String, Table, delete, insert, select

from langcheck.database.schema import METADATA
from langcheck.dictionary.words import MAX_WORD_LENGTH, clean_words

DICTIONARY_WORDS = Table(
    "dictionary_words",
    METADATA,
    Column("user", String(200), primary_key=True),
    Column("uri", String(2000), primary_key=True),
    Column("word", String(MAX_WORD_LENGTH), primary_key=True),
)


class SqlDictionaryStore:
    """One row per (user, URI, word). Calls block; run them off the event loop."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def words(self, user: str, uri: str) -> frozenset[str]:
        """The document's words."""
        query = select(DICTIONARY_WORDS.c.word).where(
            DICTIONARY_WORDS.c.user == user, DICTIONARY_WORDS.c.uri == uri
        )
        with self._engine.connect() as connection:
            found: list[object] = list(connection.execute(query).scalars())
        return frozenset(word for word in found if isinstance(word, str))

    def add(self, user: str, uri: str, words: Collection[str]) -> None:
        """Add ``words``; words already there are ignored."""
        wanted = clean_words(words)
        if not wanted:
            return
        existing = select(DICTIONARY_WORDS.c.word).where(
            DICTIONARY_WORDS.c.user == user,
            DICTIONARY_WORDS.c.uri == uri,
            DICTIONARY_WORDS.c.word.in_(sorted(wanted)),
        )
        with self._engine.begin() as connection:
            present: list[object] = list(connection.execute(existing).scalars())
            rows = [
                {"user": user, "uri": uri, "word": word}
                for word in sorted(wanted - {w for w in present if isinstance(w, str)})
            ]
            if rows:
                connection.execute(insert(DICTIONARY_WORDS), rows)

    def remove(self, user: str, uri: str, words: Collection[str]) -> None:
        """Remove ``words``."""
        stripped = [w.strip() for w in words]
        with self._engine.begin() as connection:
            connection.execute(
                delete(DICTIONARY_WORDS).where(
                    DICTIONARY_WORDS.c.user == user,
                    DICTIONARY_WORDS.c.uri == uri,
                    DICTIONARY_WORDS.c.word.in_(stripped),
                )
            )

    def rename(self, user: str, old_uri: str, new_uri: str) -> None:
        """Move the words to ``new_uri``, merging with words already there."""
        moved = self.words(user, old_uri)
        with self._engine.begin() as connection:
            connection.execute(
                delete(DICTIONARY_WORDS).where(
                    DICTIONARY_WORDS.c.user == user, DICTIONARY_WORDS.c.uri == old_uri
                )
            )
        self.add(user, new_uri, moved)
