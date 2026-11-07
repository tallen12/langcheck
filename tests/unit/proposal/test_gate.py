from pathlib import Path

import pytest
from pydantic import ValidationError

from langcheck.correction.models import AbstainGate
from langcheck.proposal.gate import GATE_FILE, load_gate, save_gate

GATE = AbstainGate(belief=0.4, reduction=0.2, flag_score=0.4, bias=-0.3, hide=-0.5, confident=0.0)


def test_a_saved_gate_loads_back(tmp_path: Path) -> None:
    path = save_gate(tmp_path, GATE)

    assert path == tmp_path / GATE_FILE
    assert load_gate(tmp_path) == GATE


def test_a_proposer_without_a_gate_has_none(tmp_path: Path) -> None:
    assert load_gate(tmp_path) is None


def test_a_gate_file_missing_a_weight_is_refused(tmp_path: Path) -> None:
    (tmp_path / GATE_FILE).write_text('{"belief": 1.0}', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_gate(tmp_path)
