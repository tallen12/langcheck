from pathlib import Path

import pytest

from langcheck.judging.models import EditJudgment, SentenceJudgment, Verdict
from langcheck.judging.store import JsonlJudgmentStore, JudgmentStoreError, store_path

EDIT = EditJudgment(
    sentence="k", start=0, end=1, correction=(), judge="j", verdict=Verdict.WRONG, reason="no"
)
SENTENCE = SentenceJudgment(sentence="k", judge="j", remaining_error=False, reason="")


def test_an_empty_store_has_no_judgments(tmp_path: Path) -> None:
    assert JsonlJudgmentStore(tmp_path / "missing.jsonl").load() == []


def test_judgments_are_appended_and_read_back_in_order(tmp_path: Path) -> None:
    path = store_path(tmp_path, "cweb-g-dev")
    store = JsonlJudgmentStore(path)

    store.add([EDIT])
    store.add([])
    store.add([SENTENCE])

    assert path == tmp_path / "cweb-g-dev" / "judgments.jsonl"
    assert JsonlJudgmentStore(path).load() == [EDIT, SENTENCE]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_line_that_is_not_a_judgment_names_its_place(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text('\n{"type": "edit"}\n', encoding="utf-8")

    with pytest.raises(JudgmentStoreError, match=r"judgments\.jsonl:2"):
        JsonlJudgmentStore(path).load()
