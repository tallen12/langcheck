from langcheck.evaluation.models import ScoredEdit
from langcheck.tagging.models import SlotScores
from langcheck.tagging.tier_report import (
    COMMA_ROW,
    ERROR_ROW,
    SUGGESTION_ROW,
    TierReport,
    TierSentence,
    measure_tiers,
    render_tiers,
)
from langcheck.tagging.tiers import TierPolicy

GO = ScoredEdit(1, 2, ("goes",), None)
COMMA = ScoredEdit(1, 1, (",",), None)
THE = ScoredEdit(1, 1, ("the",), None)
POLICY = TierPolicy(error=0.7, suggestion=0.3, comma=0.5)


def report(sentences: list[TierSentence]) -> TierReport:
    return measure_tiers(sentences, dataset="set", policy=POLICY)


def rows(result: TierReport) -> dict[str, tuple[int, int]]:
    return {row.name: (row.flags, row.correct) for row in result.rows}


def test_flags_count_under_their_tier_and_against_gold() -> None:
    result = report(
        [
            TierSentence(SlotScores((0.0, 0.9), (0.0, 0.0, 0.0)), None, ((GO,),)),
            TierSentence(SlotScores((0.5, 0.0), (0.0, 0.0, 0.0)), None, ((),)),
        ]
    )

    assert rows(result) == {ERROR_ROW: (1, 1), SUGGESTION_ROW: (1, 0), COMMA_ROW: (0, 0)}
    assert result.clean_sentences == 1
    assert result.edits == 1


def test_coverage_grows_as_more_classes_are_shown() -> None:
    result = report(
        [
            TierSentence(SlotScores((0.0, 0.5), (0.0, 0.0, 0.0)), None, ((GO,),)),
            TierSentence(SlotScores((0.5, 0.0), (0.0, 0.0, 0.0)), None, ((),)),
        ]
    )

    errors, suggestions, _ = result.coverage
    assert (errors.edit_recall, errors.clean_sentences_flagged) == (0.0, 0.0)
    assert (suggestions.edit_recall, suggestions.clean_sentences_flagged) == (1.0, 1.0)
    assert suggestions.flags_per_100_sentences == 100.0


def test_a_merged_flag_is_one_flag_but_several_unmerged_slots() -> None:
    result = report([TierSentence(SlotScores((0.0, 0.9), (0.0, 0.5, 0.0)), None, ((GO,),))])

    assert rows(result)[ERROR_ROW] == (1, 1)
    assert result.unmerged_slots == 2
    assert result.unmerged_correct == 1


def test_a_sentence_counts_against_the_annotator_its_flags_agree_with() -> None:
    other = ScoredEdit(0, 1, ("x",), None)
    result = report(
        [TierSentence(SlotScores((0.0, 0.9), (0.0, 0.0, 0.0)), None, ((other,), (GO,)))]
    )

    assert rows(result)[ERROR_ROW] == (1, 1)


def test_comma_calls_are_compared_with_what_gold_inserted() -> None:
    scores = SlotScores((0.0, 0.0), (0.0, 0.9, 0.0))
    result = report(
        [
            TierSentence(scores, (0.0, 0.9, 0.0), ((COMMA,),)),
            TierSentence(scores, (0.0, 0.9, 0.0), ((THE,),)),
            TierSentence(scores, (0.0, 0.1, 0.0), ((THE,),)),
        ]
    )

    assert rows(result)[COMMA_ROW] == (2, 2)
    assert (result.gold_comma_gaps_flagged, result.gold_comma_gaps_called_comma) == (1, 1)
    assert (result.other_gold_gaps_flagged, result.other_gold_gaps_called_comma) == (2, 1)


def test_the_table_shows_each_class_and_survives_saving() -> None:
    result = report([TierSentence(SlotScores((0.9,), (0.0, 0.0)), None, ((),))])

    table = render_tiers(result)

    assert "error" in table
    assert "errors+suggestions" in table
    assert TierReport.model_validate_json(result.model_dump_json()) == result
