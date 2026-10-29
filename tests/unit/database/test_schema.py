from pathlib import Path

from sqlalchemy import inspect

from langcheck.auth.sql_store import API_KEYS
from langcheck.database.schema import METADATA, open_database


def test_open_database_creates_every_imported_table(tmp_path: Path) -> None:
    engine = open_database(f"sqlite:///{tmp_path / 'db.sqlite'}")

    assert API_KEYS.name in METADATA.tables
    assert API_KEYS.name in inspect(engine).get_table_names()
