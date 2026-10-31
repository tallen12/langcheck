from collections.abc import Sequence

from langcheck.tagging.models import SlotScores
from langcheck.tagging.protocols import SlotScorer


class Silent:
    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        return [SlotScores((0.0,) * len(s), (0.0,) * (len(s) + 1)) for s in sentences]


def test_a_scorer_gives_one_result_per_sentence() -> None:
    scorer: SlotScorer = Silent()

    assert [len(result.gaps) for result in scorer.score([("a", "b"), ()])] == [3, 1]
