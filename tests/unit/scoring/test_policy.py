from langcheck.scoring.policy import OFF, PLL_DEFAULT, KindPolicy, ScoringPolicy
from langcheck.taxonomy.kinds import ErrorKind


def test_every_kind_is_unfiltered_by_default() -> None:
    policy = ScoringPolicy()

    assert all(policy.for_kind(kind) == OFF for kind in ErrorKind)
    assert policy.keeps(ErrorKind.SPELLING, -100.0)


def test_filtered_kinds_need_an_option_that_beats_the_margin() -> None:
    policy = ScoringPolicy({ErrorKind.PUNCTUATION: KindPolicy(filter=True, margin=0.5)})

    assert policy.keeps(ErrorKind.PUNCTUATION, 0.6)
    assert not policy.keeps(ErrorKind.PUNCTUATION, 0.5)
    assert policy.keeps(ErrorKind.SPELLING, -1.0)


def test_the_pll_default_filters_the_closeness_tuned_kinds() -> None:
    filtered = {kind for kind in ErrorKind if PLL_DEFAULT.for_kind(kind).filter}

    assert filtered == {ErrorKind.SPELLING, ErrorKind.WORD_BOUNDARY, ErrorKind.WORD_ORDER}
    # The tuned margins themselves are dropped, as tuning measured.
    assert not PLL_DEFAULT.keeps(ErrorKind.SPELLING, -0.38206120332082083)
    assert not PLL_DEFAULT.keeps(ErrorKind.WORD_BOUNDARY, 0.3304743731305715)
    assert not PLL_DEFAULT.keeps(ErrorKind.WORD_ORDER, -0.23164437233323332)
    assert PLL_DEFAULT.keeps(ErrorKind.SPELLING, -0.3)
    assert PLL_DEFAULT.keeps(ErrorKind.WORD_BOUNDARY, 0.34)
    assert PLL_DEFAULT.keeps(ErrorKind.WORD_ORDER, -0.2)
    assert PLL_DEFAULT.keeps(ErrorKind.PUNCTUATION, -1.0)
