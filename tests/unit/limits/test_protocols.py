from langcheck.limits.models import Admission, RateLimited
from langcheck.limits.protocols import CheckLimiter


class Closed:
    def admit(self, user: str) -> Admission:
        return RateLimited(1.0)


def test_check_limiter_contract() -> None:
    limiter: CheckLimiter = Closed()

    assert limiter.admit("ana") == RateLimited(1.0)
