import pytest

from langcheck.correction.models import (
    Candidate,
    CorrectionPolicy,
    FlagType,
    Ranked,
    flag_type,
)
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier
from langcheck.taxonomy.kinds import ErrorKind, Operation


def flag(start: int, end: int, kind: FlagKind = FlagKind.GENERAL) -> TextFlag:
    return TextFlag(CharSpan(start, end), 0.9, Tier.ERROR, kind)


def test_a_flag_on_text_is_wrong_text() -> None:
    assert flag_type(flag(1, 3)) is FlagType.WRONG_TEXT


def test_a_flag_on_nothing_is_missing_text() -> None:
    assert flag_type(flag(3, 3)) is FlagType.MISSING_TEXT


def test_an_optional_comma_keeps_its_type() -> None:
    assert flag_type(flag(3, 3, FlagKind.OPTIONAL_COMMA)) is FlagType.OPTIONAL_COMMA


def test_policy_needs_a_correction_to_ask_for() -> None:
    with pytest.raises(ValueError, match="count"):
        CorrectionPolicy(count=0)


def test_policy_refuses_a_negative_reduction() -> None:
    with pytest.raises(ValueError, match="min_reduction"):
        CorrectionPolicy(min_reduction=-0.1)


def test_reduction_is_the_score_lost() -> None:
    candidate = Candidate(CharSpan(0, 1), "a", "b", ErrorKind.SPELLING, Operation.REPLACEMENT, "")

    assert Ranked(candidate, 1.5, 0.5).reduction == 1.0
