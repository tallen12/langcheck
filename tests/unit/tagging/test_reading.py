from langcheck.tagging.reading import PairedReader
from tests.conftest import WordTableTagger
from tests.unit.tagging.test_document import NoCommas


def test_without_commas_a_reading_has_none() -> None:
    [reading] = PairedReader(WordTableTagger({"go": 0.9})).read([("go",)])

    assert reading.scores.words == (0.9,)
    assert reading.commas is None


def test_both_scorers_read_the_same_contexts() -> None:
    commas = NoCommas()

    [reading] = PairedReader(WordTableTagger({}), commas).read([("go",)], contexts=["Before."])

    assert reading.commas == (0.0, 0.0)
    assert commas.contexts == ["Before."]
