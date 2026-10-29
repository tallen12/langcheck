"""Protocols of the auth domain."""

from collections.abc import Mapping
from typing import Protocol

from langcheck.auth.models import AuthResult, User


class Authenticator(Protocol):
    """Identifies the user behind a request from its headers."""

    async def authenticate(self, headers: Mapping[str, str]) -> AuthResult:
        """The user, or ``Unauthenticated``. Header names are lowercase."""
        ...


class ApiKeyStore(Protocol):
    """Stores the hash of one API key per user; never the key itself."""

    def user_for(self, token_hash: str) -> User | None:
        """The user whose key has ``token_hash``, or ``None``."""
        ...

    def add(self, name: str, token_hash: str) -> None:
        """Give ``name`` the key with ``token_hash``; ``UserExistsError`` if it has one."""
        ...

    def revoke(self, name: str) -> bool:
        """Remove ``name``'s key; ``False`` when there was none."""
        ...

    def users(self) -> list[str]:
        """The names of every user with a key, sorted."""
        ...
