from pathlib import Path

from langcheck.database.schema import open_database
from langcheck.dictionary.sql_store import SqlDictionaryStore
from tests.unit.dictionary.test_memory_store import A, check_contract


def test_sql_store_contract(tmp_path: Path) -> None:
    check_contract(SqlDictionaryStore(open_database(f"sqlite:///{tmp_path / 'db.sqlite'}")))


def test_words_survive_reopening(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    SqlDictionaryStore(open_database(url)).add("ana", A, ["Bilbo"])

    assert SqlDictionaryStore(open_database(url)).words("ana", A) == {"Bilbo"}
