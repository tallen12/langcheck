"""Who made a request, or why nobody could be identified."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class User:
    """An authenticated user. ``name`` scopes caches, documents and dictionaries."""

    name: str


@dataclass(frozen=True, slots=True)
class Unauthenticated:
    """The request did not identify a user; ``reason`` is safe to show the client."""

    reason: str


type AuthResult = User | Unauthenticated

ANONYMOUS = User("anonymous")
"""The one user of a server that runs without authentication."""


class UserExistsError(Exception):
    """A user with that name already has an API key."""
