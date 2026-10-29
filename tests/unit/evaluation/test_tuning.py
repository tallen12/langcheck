from langcheck.evaluation.report import Outcome
from langcheck.evaluation.runner import Effect
from langcheck.evaluation.scoring import Counts
from langcheck.evaluation.tuning import KindDecision, Verdict, describe, policy_yaml, tune
from langcheck.scoring.policy import KindPolicy, ScoringPolicy
from langcheck.taxonomy.kinds import ErrorKind

PUNCT = ErrorKind.PUNCTUATION
OVERALL = Counts(tp=10, fp=10, fn=20)


def outcome(kind: ErrorKind, delta: float | None, correct: bool) -> Outcome:
    return Outcome(kind=kind, delta=delta, effect=Effect.CORRECT if correct else Effect.WRONG)


def test_filters_a_kind_whose_low_deltas_are_wrong_judged_on_overall_f05() -> None:
    outcomes = [
        outcome(PUNCT, -1.0, False),
        outcome(PUNCT, -0.5, False),
        outcome(PUNCT, 0.2, False),
        outcome(PUNCT, 1.0, True),
        outcome(PUNCT, 2.0, True),
    ]

    [decision] = tune(outcomes, OVERALL, min_suggestions=5)

    assert decision == KindDecision(
        kind=PUNCT,
        verdict=Verdict.FILTER,
        margin=0.2,
        suggestions=5,
        correct=2,
        dropped_correct=0,
        dropped_wrong=3,
        before=OVERALL,
        after=Counts(tp=10, fp=7, fn=20),
    )
    assert decision.filter


def test_never_drops_more_correct_suggestions_than_the_recall_limit() -> None:
    # Dropping both low deltas would help F0.5 but loses half of the correct ones.
    outcomes = [outcome(PUNCT, -1.0, True), outcome(PUNCT, -0.9, False), outcome(PUNCT, 1.0, True)]

    [decision] = tune(outcomes, OVERALL, max_recall_loss=0.1, min_suggestions=1)

    assert decision.verdict is Verdict.NO_GAIN
    assert (decision.dropped_correct, decision.dropped_wrong) == (0, 0)
    assert decision.after == decision.before


def test_kinds_with_too_few_scored_suggestions_are_not_tuned() -> None:
    outcomes = [outcome(PUNCT, -1.0, False), outcome(PUNCT, 1.0, True)]

    [decision] = tune(outcomes, OVERALL)
    [tuned] = tune(outcomes, OVERALL, min_suggestions=2)

    assert decision == KindDecision(PUNCT, Verdict.TOO_FEW, 0.0, 2, 1, 0, 0, OVERALL, OVERALL)
    assert tuned.filter


def test_unscored_suggestions_and_kinds_without_scores_are_skipped() -> None:
    outcomes = [outcome(PUNCT, None, False), outcome(ErrorKind.SPELLING, 0.5, True)]

    decisions = tune(outcomes, OVERALL, min_suggestions=1)

    assert [d.kind for d in decisions] == [ErrorKind.SPELLING]
    assert not decisions[0].filter


def test_the_margin_drops_exactly_the_suggestions_tuning_measured() -> None:
    # The wrong suggestion's delta has more decimals than a rounded margin would keep.
    outcomes = [outcome(PUNCT, 0.12344, False), outcome(PUNCT, 0.5, True)]

    [decision] = tune(outcomes, Counts(tp=1, fp=1, fn=0), min_suggestions=2)

    policy = ScoringPolicy({PUNCT: KindPolicy(filter=decision.filter, margin=decision.margin)})
    assert decision.filter
    assert not policy.keeps(PUNCT, 0.12344)
    assert policy.keeps(PUNCT, 0.5)


def filtering(margin: float) -> KindDecision:
    return KindDecision(PUNCT, Verdict.FILTER, margin, 29, 1, 0, 24, OVERALL, OVERALL)


def test_policy_yaml_lists_only_filtered_kinds_with_exact_margins() -> None:
    off = KindDecision(ErrorKind.SPELLING, Verdict.NO_GAIN, 0.0, 30, 9, 0, 0, OVERALL, OVERALL)

    assert policy_yaml([filtering(1.5099225401878353), off]) == (
        "scorer:\n  policy:\n    Punctuation:\n      filter: true\n"
        "      margin: 1.5099225401878353\n"
    )
    assert policy_yaml([off]) == "scorer:\n  policy: {}\n"


def test_describe_rounds_the_margin_for_reading_only() -> None:
    line = describe(filtering(1.5099225401878353))

    assert line.split()[:4] == ["Punctuation", "filter", "above", "1.510"]
    assert "29 suggestions,   1 correct, drops  24 wrong and  0 correct" in line
    too_few = KindDecision(PUNCT, Verdict.TOO_FEW, 0.0, 3, 0, 0, 0, OVERALL, OVERALL)
    assert "too few to tune" in describe(too_few)


def test_the_yaml_loads_into_the_server_settings() -> None:
    import yaml

    from langcheck.settings.loading import ServerSettings, load_settings

    data: object = yaml.safe_load(policy_yaml([filtering(0.2)]))
    assert isinstance(data, dict)

    settings = load_settings(ServerSettings, overrides={}, yaml_data=data)

    policy = settings.scorer.scorer_config().policy.for_kind(PUNCT)
    assert (policy.filter, policy.margin) == (True, 0.2)


def test_suggestions_with_no_effect_are_ignored() -> None:
    # Spacing "fixes" in tokenized input make no edit: dropping them changes nothing.
    no_effect = [Outcome(kind=PUNCT, delta=-1.0, effect=Effect.NONE)] * 30
    right = [outcome(PUNCT, 1.0, True)]

    [decision] = tune([*no_effect, *right], OVERALL, min_suggestions=1)

    assert (decision.suggestions, decision.correct) == (1, 1)
    assert decision.verdict is Verdict.NO_GAIN
