"""The abstain gate saved beside a proposer.

A gate's weights are fitted on one proposer's beliefs, so they live in that proposer's
directory as ``gate.json`` rather than in the code. A proposer without one offers every
correction that passes the other filters.
"""

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from langcheck.correction.models import AbstainGate

GATE_FILE: Final = "gate.json"


class GateWeights(BaseModel):
    """``gate.json``: the logistic model's weights and its two thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    belief: float
    reduction: float
    flag_score: float
    bias: float
    hide: float
    confident: float

    def gate(self) -> AbstainGate:
        """The gate these weights describe."""
        return AbstainGate(
            belief=self.belief,
            reduction=self.reduction,
            flag_score=self.flag_score,
            bias=self.bias,
            hide=self.hide,
            confident=self.confident,
        )


def load_gate(proposer: Path) -> AbstainGate | None:
    """The gate saved with the proposer at ``proposer``; None when it has none."""
    path = proposer / GATE_FILE
    if not path.is_file():
        return None
    return GateWeights.model_validate_json(path.read_text(encoding="utf-8")).gate()


def save_gate(proposer: Path, gate: AbstainGate) -> Path:
    """Write ``gate`` beside the proposer at ``proposer``; where it went."""
    weights = GateWeights(
        belief=gate.belief,
        reduction=gate.reduction,
        flag_score=gate.flag_score,
        bias=gate.bias,
        hide=gate.hide,
        confident=gate.confident,
    )
    path = proposer / GATE_FILE
    path.write_text(weights.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
