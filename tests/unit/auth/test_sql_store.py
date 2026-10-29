from datetime import UTC, datetime
from pathlib import Path

import pytest

from langcheck.auth.models import User, UserExistsError
from langcheck.auth.sql_store import SqlApiKeyStore
from langcheck.database.schema import open_database

NOW = datetime(2026, 10, 28, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> SqlApiKeyStore:
    engine = open_database(f"sqlite:///{tmp_path / 'langcheck.db'}")
    return SqlApiKeyStore(engine, clock=lambda: NOW)


def test_adds_finds_lists_and_revokes_keys(store: SqlApiKeyStore) -> None:
    store.add("ben", "h-ben")
    store.add("ana", "h-ana")

    assert store.user_for("h-ana") == User("ana")
    assert store.user_for("h-unknown") is None
    assert store.users() == ["ana", "ben"]
    assert store.revoke("ana") is True
    assert store.revoke("ana") is False
    assert store.user_for("h-ana") is None


def test_one_key_per_user(store: SqlApiKeyStore) -> None:
    store.add("ana", "h1")

    with pytest.raises(UserExistsError, match="revoke it first"):
        store.add("ana", "h2")
    assert store.user_for("h1") == User("ana")


def test_keys_survive_reopening_the_database(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'langcheck.db'}"
    SqlApiKeyStore(open_database(url), clock=lambda: NOW).add("ana", "h1")

    assert SqlApiKeyStore(open_database(url), clock=lambda: NOW).users() == ["ana"]
