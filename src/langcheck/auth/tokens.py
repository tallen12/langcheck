"""Creating API keys, hashing them, and managing users' keys."""

import hashlib
import secrets
from collections.abc import Callable

from langcheck.auth.protocols import ApiKeyStore

TOKEN_PREFIX = "lc_"
TOKEN_BYTES = 32


def new_token(*, random: Callable[[int], str] = secrets.token_urlsafe) -> str:
    """A new API key with a recognisable prefix, so leaked keys are easy to scan for."""
    return TOKEN_PREFIX + random(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """The stored form of ``token``. Keys are long and random, so a plain SHA-256 is
    enough; a slow password hash would only slow down every request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def add_user(
    store: ApiKeyStore, name: str, *, random: Callable[[int], str] = secrets.token_urlsafe
) -> str:
    """Create a key for ``name`` and return it; only its hash is stored."""
    if not name.strip():
        msg = "user name must not be blank"
        raise ValueError(msg)
    token = new_token(random=random)
    store.add(name.strip(), hash_token(token))
    return token
