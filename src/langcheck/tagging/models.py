"""Error probabilities for the slots of a sentence."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlotScores:
    """The probability of an error at each slot of a sentence of ``n`` words.

    ``words`` has ``n`` entries: this word is wrong. ``gaps`` has ``n + 1``: something is
    missing before word ``i`` (the last gap is after the final word). Slots past the
    model's length limit score 0.
    """

    words: tuple[float, ...]
    gaps: tuple[float, ...]

    @property
    def highest(self) -> float:
        """The sentence's score: the probability of its most likely error."""
        return max((*self.words, *self.gaps))


@dataclass(frozen=True, slots=True)
class SentenceReading:
    """All a tagger says about a sentence. ``commas`` is each gap's probability that what is
    missing there is a comma; ``None`` for a tagger without a comma head.
    """

    scores: SlotScores
    commas: tuple[float, ...] | None
