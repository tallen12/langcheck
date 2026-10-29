import asyncio
from collections.abc import Mapping

from langcheck.auth.models import AuthResult, User
from langcheck.auth.protocols import Authenticator


class HeaderAuthenticator:
    """Names the user after the ``x-user`` header."""

    async def authenticate(self, headers: Mapping[str, str]) -> AuthResult:
        return User(headers["x-user"])


def test_authenticator_contract() -> None:
    authenticator: Authenticator = HeaderAuthenticator()

    assert asyncio.run(authenticator.authenticate({"x-user": "ana"})) == User("ana")
