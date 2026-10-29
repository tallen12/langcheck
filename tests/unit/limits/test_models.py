from langcheck.limits.models import Permit


def test_a_permit_releases_once() -> None:
    released: list[int] = []
    permit = Permit(lambda: released.append(1))

    with permit:
        pass
    permit.release()

    assert released == [1]
