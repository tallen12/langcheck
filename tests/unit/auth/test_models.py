from langcheck.auth.models import ANONYMOUS, Unauthenticated, User


def test_users_compare_by_name() -> None:
    assert User("ana") == User("ana")
    assert ANONYMOUS == User("anonymous")
    assert Unauthenticated("no key").reason == "no key"
