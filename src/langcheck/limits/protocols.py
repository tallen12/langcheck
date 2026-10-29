"""Protocols of the limits domain."""

from typing import Protocol

from langcheck.limits.models import Admission


class CheckLimiter(Protocol):
    """Decides whether a user may start another check now."""

    def admit(self, user: str) -> Admission:
        """A ``Permit`` to hold while checking, or ``RateLimited``."""
        ...
