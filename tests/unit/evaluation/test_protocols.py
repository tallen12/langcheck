from collections.abc import Sequence

from langcheck.evaluation.m2 import GoldEdit
from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.protocols import EditAnnotator


class NoEdits:
    """An annotator that never finds anything to correct."""

    def system_edits(self, source: Sequence[str], corrected: str) -> list[ScoredEdit]:
        return []

    def reference_edits(self, source: Sequence[str], reference: Sequence[str]) -> list[ScoredEdit]:
        return []

    def gold_edits(self, source: Sequence[str], edits: Sequence[GoldEdit]) -> list[ScoredEdit]:
        return [ScoredEdit(e.start, e.end, e.correction, None) for e in edits]


def test_edit_annotator_contract() -> None:
    annotator: EditAnnotator = NoEdits()

    assert annotator.system_edits(("a",), "a") == []
    assert annotator.gold_edits(("a",), [GoldEdit(0, 1, ("b",), "R:OTHER")])[0].correction == ("b",)
