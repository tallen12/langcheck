from langcheck.tagging.models import SlotScores


def test_a_sentence_scores_as_its_most_likely_error() -> None:
    assert SlotScores((0.1, 0.2), (0.0, 0.7, 0.3)).highest == 0.7
