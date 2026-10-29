"""The built-in authentication modes: ``api_key`` and ``none``."""

import asyncio
from collections.abc import Mapping

from langcheck.auth.models import ANONYMOUS, AuthResult, Unauthenticated, User
from langcheck.auth.protocols import ApiKeyStore
from langcheck.auth.tokens import hash_token

AUTHORIZATION = "authorization"
BEARER = "bearer"


class ApiKeyAuthenticator:
    """Accepts ``Authorization: Bearer <key>`` for a key in the store."""

    def __init__(self, store: ApiKeyStore) -> None:
        self._store = store

    async def authenticate(self, headers: Mapping[str, str]) -> AuthResult:
        """The key's user, or ``Unauthenticated`` for a missing or unknown key."""
        scheme, _, token = headers.get(AUTHORIZATION, "").strip().partition(" ")
        if scheme.lower() != BEARER or not token.strip():
            return Unauthenticated("missing bearer token")
        # The store may block (SQL), so it runs in a worker thread.
        user = await asyncio.to_thread(self._store.user_for, hash_token(token.strip()))
        return user if user is not None else Unauthenticated("invalid API key")


class TrustingAuthenticator:
    """Mode ``none``: trusts every request.

    Without ``proxy_user_header`` every request belongs to :data:`ANONYMOUS`. With it, a
    reverse proxy that has already authenticated the user names them in that header, and
    a request without it is refused, since it did not come through the proxy.
    """

    def __init__(self, *, proxy_user_header: str | None) -> None:
        self._header = proxy_user_header.lower() if proxy_user_header is not None else None

    async def authenticate(self, headers: Mapping[str, str]) -> AuthResult:
        """The anonymous user, or the user the proxy named."""
        if self._header is None:
            return ANONYMOUS
        name = headers.get(self._header, "").strip()
        return User(name) if name else Unauthenticated(f"missing {self._header} header")
