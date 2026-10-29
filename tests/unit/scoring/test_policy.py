from langcheck.scoring.policy import OFF, KindPolicy, ScoringPolicy
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
