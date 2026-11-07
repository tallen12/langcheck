from langcheck.judging.clean import (
    CleanLine,
    CleanSentence,
    LeftOut,
    LeftOutReason,
    Scope,
    clean,
    clean_line,
)
from langcheck.judging.consensus import Agreed, Consensus, Disputed, Pending
from langcheck.judging.models import Candidate, EditKey, Item, Source, Verdict, Version
from langcheck.judging.panel import ItemConsensus

TOKENS = ("He", "go", "to", "school", ".")
GOES: EditKey = (1, 2, ("goes",))
WENT: EditKey = (1, 2, ("went",))
THE: EditKey = (3, 3, ("the",))
NO: Consensus[bool] = Agreed(value=False, votes=3)


def decided(
    versions: dict[str, tuple[EditKey, ...]],
    verdicts: dict[EditKey, Consensus[Verdict]],
    *,
    linked: frozenset[EditKey] = frozenset(),
    remaining: Consensus[bool] = NO,
    preference: Consensus[str] | None = None,
) -> ItemConsensus:
    candidates = tuple(Candidate(*key, frozenset({Source.GOLD})) for key in sorted(verdicts))
    item = Item(
        TOKENS,
        "",
        candidates,
        tuple(Version(label, edits) for label, edits in versions.items()),
        "He go to school.",
    )
    return ItemConsensus(item, verdicts, linked, remaining, preference)


def valid() -> Agreed[Verdict]:
    return Agreed(value=Verdict.VALID, votes=2)


def test_keeps_only_valid_edits_in_published_spacing() -> None:
    result = clean(
        decided(
            {"annotator 1": (GOES, THE)},
            {GOES: valid(), THE: Agreed(value=Verdict.NEUTRAL, votes=2)},
        ),
        scope=Scope.LOCAL,
    )

    assert isinstance(result, CleanSentence)
    assert result.text == "He go to school."
    assert result.references == ("He goes to school.",)
    assert not result.linked


def test_a_sentence_without_edits_is_kept_unchanged() -> None:
    result = clean(decided({}, {}), scope=Scope.LOCAL)

    assert isinstance(result, CleanSentence)
    assert result.references == ("He go to school.",)


def test_the_preferred_version_is_the_main_reference() -> None:
    result = clean(
        decided(
            {"annotator 1": (GOES,), "annotator 2": (WENT,)},
            {GOES: valid(), WENT: valid()},
            preference=Agreed(value="annotator 2", votes=2),
        ),
        scope=Scope.LOCAL,
    )

    assert isinstance(result, CleanSentence)
    assert result.references == ("He went to school.", "He goes to school.")


def test_without_a_preference_the_version_with_most_valid_edits_leads() -> None:
    result = clean(
        decided(
            {"annotator 1": (GOES,), "annotator 2": (WENT, THE)},
            {GOES: valid(), WENT: valid(), THE: valid()},
            preference=Agreed(value="tie", votes=2),
        ),
        scope=Scope.LOCAL,
    )

    assert isinstance(result, CleanSentence)
    assert result.references[0] == "He went to the school."


def test_a_version_with_every_edit_rejected_is_not_a_reference() -> None:
    result = clean(
        decided(
            {"annotator 1": (GOES,), "annotator 2": (THE,)},
            {GOES: valid(), THE: Agreed(value=Verdict.WRONG, votes=3)},
        ),
        scope=Scope.LOCAL,
    )

    assert isinstance(result, CleanSentence)
    assert result.references == ("He goes to school.",)


def test_unsettled_sentences_are_left_out_with_the_reason() -> None:
    cases = [
        (decided({"a": (GOES,)}, {GOES: Pending()}), LeftOutReason.PENDING),
        (decided({"a": (GOES,)}, {GOES: Disputed()}), LeftOutReason.DISPUTED),
        (decided({}, {}, remaining=Disputed()), LeftOutReason.DISPUTED),
        (decided({}, {}, remaining=Agreed(value=True, votes=2)), LeftOutReason.ERROR_LEFT),
    ]

    for case, reason in cases:
        assert clean(case, scope=Scope.ALL) == LeftOut(case.item.key, reason)


def test_linked_edits_are_left_out_locally_and_marked_in_full() -> None:
    case = decided({"a": (GOES, THE)}, {GOES: valid(), THE: valid()}, linked=frozenset({THE}))

    assert clean(case, scope=Scope.LOCAL) == LeftOut(case.item.key, LeftOutReason.NON_LOCAL)
    full = clean(case, scope=Scope.ALL)
    assert isinstance(full, CleanSentence)
    assert full.linked


def test_a_rejected_linked_edit_does_not_leave_the_sentence_out() -> None:
    case = decided(
        {"a": (GOES, THE)},
        {GOES: valid(), THE: Agreed(value=Verdict.NEUTRAL, votes=2)},
        linked=frozenset({THE}),
    )

    assert isinstance(clean(case, scope=Scope.LOCAL), CleanSentence)


def test_clean_line_reads_as_a_corrections_file() -> None:
    line = clean_line(CleanSentence("k", "He go.", ("He goes.", "He went."), linked=False))

    assert line == CleanLine(
        sentence="k", text="He go.", corrected="He goes.", alternatives=("He went.",)
    )
