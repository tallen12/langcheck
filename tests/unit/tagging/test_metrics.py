from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.scoring import Counts
from langcheck.tagging.metrics import Detection, ScoredSentence, measure, operating_point
from langcheck.tagging.models import SlotScores
from langcheck.taxonomy.kinds import ErrorKind

VERB = ScoredEdit(1, 2, ("goes",), ErrorKind.VERB_TENSE)
MISSING_THE = ScoredEdit(2, 2, ("the",), ErrorKind.DETERMINER)


def test_slots_edits_and_sentences_are_counted_at_the_threshold() -> None:
    wrong = ScoredSentence(
        SlotScores((0.1, 0.9, 0.8), (0.0, 0.0, 0.1, 0.0)), ((VERB, MISSING_THE),)
    )
    clean = ScoredSentence(SlotScores((0.1,), (0.0, 0.0)), ((),))
    false_alarm = ScoredSentence(SlotScores((0.9,), (0.0, 0.0)), ((),))

    result = measure([wrong, clean, false_alarm], 0.5)

    assert result.slots == Counts(tp=1, fp=2, fn=1)
    assert result.sentences == Counts(tp=1, fp=1, fn=0)
    assert result.edits == Detection(1, 2)
    assert result.by_kind == {
        ErrorKind.VERB_TENSE: Detection(1, 1),
        ErrorKind.DETERMINER: Detection(0, 1),
    }


def test_a_multi_word_edit_is_found_by_a_flag_on_any_of_its_words() -> None:
    edit = ScoredEdit(0, 2, ("x",), None)
    sentence = ScoredSentence(SlotScores((0.0, 0.9), (0.0, 0.0, 0.0)), ((edit,),))

    assert measure([sentence], 0.5).by_kind[None] == Detection(1, 1)


def test_a_sentence_counts_against_its_best_annotator() -> None:
    other = ScoredEdit(0, 1, ("x",), None)
    sentence = ScoredSentence(SlotScores((0.0, 0.9), (0.0, 0.0, 0.0)), ((other,), (VERB,)))

    assert measure([sentence], 0.5).slots == Counts(tp=1)


def test_the_operating_point_has_the_best_recall_at_the_precision() -> None:
    sentences = [
        ScoredSentence(SlotScores((0.9, 0.6, 0.3), (0.0,) * 4), ((VERB,),)),
        ScoredSentence(SlotScores((0.7,), (0.0, 0.0)), ((ScoredEdit(0, 1, (), None),),)),
    ]
    # Word 0 of the first sentence is a false alarm up to 0.90; word 2 would be another.
    point = operating_point(sentences, precision=0.6)

    assert point.slots == Counts(tp=2, fp=1)
    assert 0.3 < point.threshold <= 0.6


def test_with_no_threshold_precise_enough_the_most_precise_is_used() -> None:
    sentence = ScoredSentence(SlotScores((0.9, 0.5), (0.0, 0.0, 0.0)), ((VERB,),))

    point = operating_point([sentence], precision=0.9)

    assert point.slots.tp == 1
    assert point.slots.precision == 0.5
