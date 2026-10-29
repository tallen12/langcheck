"""What a sentence scorer rates."""

from dataclasses import dataclass

from langcheck.text.models import Span


@dataclass(frozen=True, slots=True)
class ScoredText:
    """A sentence to score, with ``focus`` on the edited span (relative to ``text``).

    Scorers that look at the whole sentence ignore ``focus``; a masked-LM scorer only
    scores the tokens around it, which is faster and compares a candidate with its
    original over the same stretch of text.
    """

    text: str
    focus: Span

    def __post_init__(self) -> None:
        if self.focus.end > len(self.text):
            msg = f"focus {self.focus} is outside a text of {len(self.text)} code points"
            raise ValueError(msg)
