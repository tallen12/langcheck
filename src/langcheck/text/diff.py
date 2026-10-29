"""Matching the sentences of two versions of a text, to see which survived unchanged."""

from dataclasses import dataclass
from difflib import SequenceMatcher

from langcheck.text.models import Segmentation, Span


@dataclass(frozen=True, slots=True)
class SentenceMove:
    """An unchanged sentence: where it was in the old text and where it is now."""

    old: Span
    new: Span

    @property
    def offset(self) -> int:
        """How far the sentence moved, in code points."""
        return self.new.start - self.old.start


@dataclass(frozen=True, slots=True)
class SentenceDiff:
    """The unchanged sentences, in order, and how many new sentences are new or edited."""

    moves: tuple[SentenceMove, ...]
    changed: int
    total: int

    @property
    def changed_share(self) -> float:
        """The share of the new text's sentences that are new or edited (0 for no text)."""
        return self.changed / self.total if self.total else 0.0

    def move_for(self, span: Span) -> SentenceMove | None:
        """The move of the unchanged old sentence that contains ``span``, if any."""
        for move in self.moves:
            if move.old.start <= span.start and span.end <= move.old.end:
                return move
        return None


def diff_sentences(old: Segmentation, new: Segmentation) -> SentenceDiff:
    """Align the sentences of ``old`` and ``new`` by their text.

    Sentences are compared whole, so an edit anywhere in a sentence makes it changed;
    moving a sentence elsewhere counts as deleting and inserting it.
    """
    old_sentences = old.sentences
    new_sentences = new.sentences
    matcher = SequenceMatcher(
        a=[s.text for s in old_sentences], b=[s.text for s in new_sentences], autojunk=False
    )
    moves = [
        SentenceMove(old_sentences[block.a + i].span, new_sentences[block.b + i].span)
        for block in matcher.get_matching_blocks()
        for i in range(block.size)
    ]
    return SentenceDiff(tuple(moves), len(new_sentences) - len(moves), len(new_sentences))
