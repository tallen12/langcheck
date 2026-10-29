import pytest

from langcheck.auth.tokens import TOKEN_PREFIX, add_user, hash_token, new_token
from tests.conftest import MemoryApiKeyStore


def test_new_tokens_are_prefixed_and_random() -> None:
    assert new_token(random=lambda n: "x" * n) == TOKEN_PREFIX + "x" * 32
    assert new_token() != new_token()


def test_hash_is_stable_hex_and_hides_the_token() -> None:
    assert hash_token("lc_abc") == hash_token("lc_abc")
    assert len(hash_token("lc_abc")) == 64
    assert "abc" not in hash_token("lc_abc")


def test_add_user_stores_only_the_hash() -> None:
    store = MemoryApiKeyStore()

    token = add_user(store, " ana ", random=lambda n: "k")

    assert token == "lc_k"
    assert store.hashes == {"ana": hash_token("lc_k")}


def test_add_user_needs_a_name() -> None:
    with pytest.raises(ValueError, match="blank"):
        add_user(MemoryApiKeyStore(), "  ")
