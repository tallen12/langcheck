import asyncio
from collections.abc import Mapping

import pytest

from langcheck.auth.authenticators import ApiKeyAuthenticator, TrustingAuthenticator
from langcheck.auth.models import ANONYMOUS, AuthResult, Unauthenticated, User
from langcheck.auth.protocols import Authenticator
from langcheck.auth.tokens import hash_token
from tests.conftest import MemoryApiKeyStore


def authenticate(authenticator: Authenticator, headers: Mapping[str, str]) -> AuthResult:
    return asyncio.run(authenticator.authenticate(headers))


@pytest.fixture
def api_keys() -> ApiKeyAuthenticator:
    store = MemoryApiKeyStore()
    store.add("ana", hash_token("lc_good"))
    return ApiKeyAuthenticator(store)


@pytest.mark.parametrize("header", ["Bearer lc_good", "bearer  lc_good ", "BEARER lc_good"])
def test_api_key_accepts_a_known_bearer_token(api_keys: ApiKeyAuthenticator, header: str) -> None:
    assert authenticate(api_keys, {"authorization": header}) == User("ana")


@pytest.mark.parametrize(
    ("headers", "reason"),
    [
        ({}, "missing bearer token"),
        ({"authorization": "Basic lc_good"}, "missing bearer token"),
        ({"authorization": "Bearer "}, "missing bearer token"),
        ({"authorization": "Bearer lc_bad"}, "invalid API key"),
    ],
)
def test_api_key_refuses_missing_or_unknown_tokens(
    api_keys: ApiKeyAuthenticator, headers: dict[str, str], reason: str
) -> None:
    assert authenticate(api_keys, headers) == Unauthenticated(reason)


def test_trusting_without_a_header_makes_everyone_anonymous() -> None:
    authenticator = TrustingAuthenticator(proxy_user_header=None)

    assert authenticate(authenticator, {}) == ANONYMOUS


def test_trusting_with_a_proxy_header_uses_the_named_user() -> None:
    authenticator = TrustingAuthenticator(proxy_user_header="X-Forwarded-User")

    assert authenticate(authenticator, {"x-forwarded-user": "ben"}) == User("ben")
    assert authenticate(authenticator, {}) == Unauthenticated("missing x-forwarded-user header")
