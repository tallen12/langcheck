import random
from collections import Counter

from langcheck.synthesis.corruption import corrupt
from langcheck.synthesis.default_tables import CODE_WEIGHTS, ERROR_COUNTS, default_confusions
from langcheck.taxonomy.errant_map import ERRANT_CODES


def test_every_code_is_one_errant_can_type() -> None:
    confusions = default_confusions()

    assert set(confusions.codes) <= ERRANT_CODES
    assert set(confusions.written) | set(confusions.inserted) <= set(CODE_WEIGHTS)


def test_tables_drop_swap_and_insert_words() -> None:
    confusions = default_confusions()

    assert confusions.written["M:DET"][("the",)] == (((),), (3,))
    assert confusions.written["R:OTHER"][("their",)] == ((("there",), ("they're",)), (2, 1))
    assert confusions.written["R:ORTH"][("a", "lot")] == ((("alot",),), (1,))
    assert confusions.inserted["U:PREP"]["discuss"] == ((("about",),), (1,))


def test_corrupted_prose_draws_its_errors_from_the_built_in_tables() -> None:
    confusions = default_confusions()
    rng = random.Random(0)
    clean = tuple("I think that their plan is to discuss the issue , and it is late .".split())

    made = [corrupt(clean, confusions, rng) for _ in range(400)]

    counts = Counter(len(sentence.annotators[0]) for sentence in made)
    assert set(counts) <= set(ERROR_COUNTS)
    assert counts[0] > 50
    codes = {edit.code for sentence in made for edit in sentence.annotators[0]}
    assert {"M:PUNCT", "R:OTHER", "U:PREP", "R:VERB:SVA"} <= codes
