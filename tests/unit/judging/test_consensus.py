import pytest

from langcheck.judging.consensus import Agreed, Disputed, Pending, consensus


def test_two_of_three_agree() -> None:
    assert consensus(["valid", "wrong", "valid"], judges=3, quorum=2) == Agreed("valid", 2)


def test_a_three_way_split_is_disputed() -> None:
    assert consensus(["valid", "wrong", "neutral"], judges=3, quorum=2) == Disputed()


def test_missing_judges_leave_it_pending_until_a_quorum_or_all_answer() -> None:
    assert consensus(["valid", "wrong"], judges=3, quorum=2) == Pending()
    assert consensus([], judges=3, quorum=2) == Pending()
    assert consensus([True, True], judges=3, quorum=2) == Agreed(True, 2)


def test_a_tie_at_a_low_quorum_is_disputed() -> None:
    assert consensus(["valid", "wrong"], judges=2, quorum=1) == Disputed()
    assert consensus(["valid", "valid", "wrong"], judges=3, quorum=1) == Agreed("valid", 2)


def test_the_quorum_must_fit_the_panel() -> None:
    with pytest.raises(ValueError, match="quorum"):
        consensus(["valid"], judges=3, quorum=4)
