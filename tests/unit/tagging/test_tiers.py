import pytest

from langcheck.tagging.alignment import CharSpan, word_spans
from langcheck.tagging.models import SlotScores
from langcheck.tagging.tiers import (
    Falloff,
    FlagKind,
    SlotFlag,
    TextFlag,
    Tier,
    TierPolicy,
    slot_flags,
    text_flag,
)

POLICY = TierPolicy(error=0.7, suggestion=0.3, comma=0.5)


def test_scores_fall_into_tiers_and_low_ones_are_dropped() -> None:
    scores = SlotScores((0.9, 0.0, 0.5, 0.2), (0.0, 0.0, 0.0, 0.0, 0.0))

    assert slot_flags(scores, None, POLICY) == [
        SlotFlag(1, 1, 0.9, Tier.ERROR, FlagKind.GENERAL),
        SlotFlag(5, 5, 0.5, Tier.SUGGESTION, FlagKind.GENERAL),
    ]


def test_a_gap_and_the_word_beside_it_are_one_flag_at_the_higher_tier() -> None:
    scores = SlotScores((0.0, 0.4, 0.0), (0.0, 0.8, 0.0, 0.0))

    assert slot_flags(scores, None, POLICY) == [SlotFlag(2, 3, 0.8, Tier.ERROR, FlagKind.GENERAL)]


def test_two_words_with_an_unflagged_gap_between_are_one_flag() -> None:
    scores = SlotScores((0.5, 0.5), (0.0, 0.0, 0.0))

    assert slot_flags(scores, None, POLICY) == [
        SlotFlag(1, 3, 0.5, Tier.SUGGESTION, FlagKind.GENERAL)
    ]


def test_two_gaps_with_an_unflagged_word_between_stay_apart() -> None:
    scores = SlotScores((0.0, 0.0), (0.5, 0.5, 0.0))

    assert [flag.first for flag in slot_flags(scores, None, POLICY)] == [0, 2]


def test_a_likely_comma_is_its_own_flag_and_splits_its_neighbours() -> None:
    scores = SlotScores((0.5, 0.5), (0.0, 0.9, 0.0))

    assert slot_flags(scores, (0.0, 0.8, 0.0), POLICY) == [
        SlotFlag(1, 1, 0.5, Tier.SUGGESTION, FlagKind.GENERAL),
        SlotFlag(2, 2, 0.9, Tier.ERROR, FlagKind.OPTIONAL_COMMA),
        SlotFlag(3, 3, 0.5, Tier.SUGGESTION, FlagKind.GENERAL),
    ]


def test_a_gap_unlikely_to_be_a_comma_stays_general() -> None:
    scores = SlotScores((0.0,), (0.9, 0.0))

    [flag] = slot_flags(scores, (0.1, 0.1), POLICY)

    assert flag.kind is FlagKind.GENERAL


def test_inverted_thresholds_are_refused() -> None:
    with pytest.raises(ValueError, match="suggestion"):
        TierPolicy(error=0.2, suggestion=0.5)


GROWING = TierPolicy(error=0.7, suggestion=0.3, comma=0.5, growth=Falloff(join=0.1, rate=2.0))
QUIET = (0.0,) * 6


def test_a_flag_takes_in_neighbours_while_they_clear_the_rising_bar() -> None:
    # The bar is 0.1 one word away and 0.2 two away: the second word on the right misses it.
    scores = SlotScores((0.12, 0.9, 0.12, 0.15, 0.0), QUIET)

    assert slot_flags(scores, None, GROWING) == [SlotFlag(1, 5, 0.9, Tier.ERROR, FlagKind.GENERAL)]


def test_a_weak_word_ends_the_flag_though_a_further_one_clears_its_bar() -> None:
    scores = SlotScores((0.9, 0.05, 0.25, 0.0), (0.0,) * 5)

    assert slot_flags(scores, None, GROWING) == [SlotFlag(1, 1, 0.9, Tier.ERROR, FlagKind.GENERAL)]


def test_weak_gaps_between_words_do_not_stop_growth() -> None:
    scores = SlotScores((0.9, 0.15), (0.0, 0.0, 0.0))

    assert slot_flags(scores, None, GROWING) == [SlotFlag(1, 3, 0.9, Tier.ERROR, FlagKind.GENERAL)]


def test_flags_join_only_across_words_likely_for_their_distance() -> None:
    apart = SlotScores((0.9, 0.05, 0.8), (0.0,) * 4)
    joined = SlotScores((0.9, 0.15, 0.8), (0.0,) * 4)

    assert [(f.first, f.last) for f in slot_flags(apart, None, GROWING)] == [(1, 1), (5, 5)]
    assert [(f.first, f.last) for f in slot_flags(joined, None, GROWING)] == [(1, 5)]


def test_weak_slots_with_no_strong_one_beside_them_are_not_flagged() -> None:
    scores = SlotScores((0.2, 0.2, 0.2), (0.0,) * 4)

    assert slot_flags(scores, None, GROWING) == []


def test_growth_stops_at_an_optional_comma() -> None:
    scores = SlotScores((0.9, 0.2), (0.0, 0.0, 0.9))

    assert slot_flags(scores, (0.0, 0.0, 0.8), GROWING) == [
        SlotFlag(1, 3, 0.9, Tier.ERROR, FlagKind.GENERAL),
        SlotFlag(4, 4, 0.9, Tier.ERROR, FlagKind.OPTIONAL_COMMA),
    ]


def test_falloff_needs_a_positive_bar_that_does_not_shrink() -> None:
    with pytest.raises(ValueError, match="join"):
        Falloff(join=0.0)
    with pytest.raises(ValueError, match="rate"):
        Falloff(join=0.1, rate=0.5)
    assert Falloff(join=0.1, rate=2.0).bar(3) == pytest.approx(0.4)


TEXT = "He go  home."
SPANS = word_spans(TEXT, ("He", "go", "home", "."))


def text_of(first: int, last: int) -> TextFlag:
    flag = SlotFlag(first, last, 0.9, Tier.ERROR, FlagKind.GENERAL)
    return text_flag(flag, SPANS)


def test_a_word_flag_covers_the_word() -> None:
    assert text_of(3, 3).span == CharSpan(3, 5)


def test_a_gap_flag_is_empty_after_the_word_before_it() -> None:
    assert text_of(4, 4).span == CharSpan(5, 5)
    assert text_of(0, 0).span == CharSpan(0, 0)
    assert text_of(8, 8).span == CharSpan(12, 12)


def test_a_merged_flag_covers_its_words() -> None:
    assert text_of(3, 5).span == CharSpan(3, 11)
    assert text_of(4, 5).span == CharSpan(5, 11)
    assert text_of(3, 4).span == CharSpan(3, 5)
