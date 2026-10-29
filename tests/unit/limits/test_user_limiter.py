import pytest

from langcheck.limits.models import Permit, RateLimited
from langcheck.limits.protocols import CheckLimiter
from langcheck.limits.user_limiter import IN_FLIGHT_RETRY, Unlimited, UserLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_limits_checks_per_minute_per_user() -> None:
    clock = Clock()
    limiter: CheckLimiter = UserLimiter(max_in_flight=5, per_minute=2, clock=clock)

    for _ in range(2):
        admitted = limiter.admit("ana")
        assert isinstance(admitted, Permit)
        admitted.release()
    clock.now += 10
    assert limiter.admit("ana") == RateLimited(50.0)
    assert isinstance(limiter.admit("ben"), Permit)
    clock.now += 50
    assert isinstance(limiter.admit("ana"), Permit)


def test_limits_checks_in_flight_until_released() -> None:
    limiter = UserLimiter(max_in_flight=1, per_minute=10, clock=Clock())

    held = limiter.admit("ana")
    assert isinstance(held, Permit)
    assert limiter.admit("ana") == RateLimited(IN_FLIGHT_RETRY)
    with held:
        pass
    held.release()
    assert isinstance(limiter.admit("ana"), Permit)


def test_needs_positive_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        UserLimiter(max_in_flight=0, per_minute=1, clock=Clock())


def test_unlimited_admits_everything() -> None:
    limiter = Unlimited()

    assert all(isinstance(limiter.admit("ana"), Permit) for _ in range(100))
