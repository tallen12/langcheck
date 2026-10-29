from langcheck.evaluation.report import Outcome
from langcheck.evaluation.scoring import Counts
from langcheck.evaluation.tuning import KindDecision, policy_yaml, tune
from langcheck.taxonomy.kinds import ErrorKind

PUNCT = ErrorKind.PUNCTUATION


def outcome(kind: ErrorKind, delta: float | None, correct: bool) -> Outcome:
    return Outcome(kind=kind, delta=delta, correct=correct)


def test_turns_filtering_on_where_low_deltas_are_mostly_wrong() -> None:
    outcomes = [
        outcome(PUNCT, -1.0, False),
        outcome(PUNCT, -0.5, False),
        outcome(PUNCT, 0.2, False),
        outcome(PUNCT, 1.0, True),
        outcome(PUNCT, 2.0, True),
    ]

    [decision] = tune(outcomes, {"Punctuation": Counts(tp=2, fp=3, fn=1)})

    assert decision == KindDecision(
        kind=PUNCT, filter=True, margin=0.2, before=Counts(2, 3, 1), after=Counts(2, 0, 1)
    )


def test_keeps_filtering_off_when_it_would_cost_too_much_recall() -> None:
    outcomes = [outcome(PUNCT, -1.0, True), outcome(PUNCT, -0.9, False), outcome(PUNCT, 1.0, True)]

    [decision] = tune(outcomes, {"Punctuation": Counts(tp=2, fp=1, fn=0)}, max_recall_loss=0.1)

    assert not decision.filter
    assert decision.after == decision.before


def test_unscored_suggestions_and_kinds_without_scores_are_skipped() -> None:
    decisions = tune([outcome(PUNCT, None, False), outcome(ErrorKind.SPELLING, 0.5, True)], {})

    assert [d.kind for d in decisions] == [ErrorKind.SPELLING]
    assert not decisions[0].filter


def test_policy_yaml_lists_only_filtered_kinds() -> None:
    on = KindDecision(PUNCT, True, 0.2, Counts(), Counts())
    off = KindDecision(ErrorKind.SPELLING, False, 0.0, Counts(), Counts())

    assert policy_yaml([on, off]) == (
        "scorer:\n  policy:\n    Punctuation:\n      filter: true\n      margin: 0.2\n"
    )
    assert policy_yaml([off]) == "scorer:\n  policy: {}\n"


def test_the_yaml_loads_into_the_server_settings() -> None:
    import yaml

    from langcheck.settings.loading import ServerSettings, load_settings

    text = policy_yaml([KindDecision(PUNCT, True, 0.2, Counts(), Counts())])
    data: object = yaml.safe_load(text)
    assert isinstance(data, dict)

    settings = load_settings(ServerSettings, overrides={}, yaml_data=data)

    policy = settings.scorer.scorer_config().policy.for_kind(PUNCT)
    assert (policy.filter, policy.margin) == (True, 0.2)
