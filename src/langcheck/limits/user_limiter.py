"""Limits on checks per user: how many at once and how many per minute."""

from collections import deque
from collections.abc import Callable

from langcheck.limits.models import Admission, Permit, RateLimited

WINDOW_SECONDS = 60.0
IN_FLIGHT_RETRY = 1.0


class UserLimiter:
    """Admits a check when the user has fewer than ``max_in_flight`` running and started
    fewer than ``per_minute`` in the last 60 seconds. A check counts as one request
    however many LLM calls its chunks need.
    """

    def __init__(self, *, max_in_flight: int, per_minute: int, clock: Callable[[], float]) -> None:
        if max_in_flight < 1 or per_minute < 1:
            msg = "max_in_flight and per_minute must be positive"
            raise ValueError(msg)
        self._max_in_flight = max_in_flight
        self._per_minute = per_minute
        self._clock = clock
        self._in_flight: dict[str, int] = {}
        self._started: dict[str, deque[float]] = {}

    def admit(self, user: str) -> Admission:
        """A ``Permit`` for one check, or how long to wait."""
        now = self._clock()
        started = self._started.setdefault(user, deque())
        while started and started[0] <= now - WINDOW_SECONDS:
            started.popleft()
        if len(started) >= self._per_minute:
            return RateLimited(started[0] + WINDOW_SECONDS - now)
        if self._in_flight.get(user, 0) >= self._max_in_flight:
            return RateLimited(IN_FLIGHT_RETRY)
        started.append(now)
        self._in_flight[user] = self._in_flight.get(user, 0) + 1
        return Permit(lambda: self._release(user))

    def _release(self, user: str) -> None:
        remaining = self._in_flight.get(user, 0) - 1
        if remaining > 0:
            self._in_flight[user] = remaining
        else:
            self._in_flight.pop(user, None)


class Unlimited:
    """Admits every check; for local runs and tests."""

    def admit(self, user: str) -> Admission:
        """Always a ``Permit``."""
        return Permit(lambda: None)
