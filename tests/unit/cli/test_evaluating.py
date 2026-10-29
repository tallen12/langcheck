from pathlib import Path

from langcheck.cli.environment import load_edit_annotator
from langcheck.cli.evaluating import DEFAULT_RUNS, _default_output
from langcheck.evaluation.datasets import Dataset
from langcheck.evaluation.protocols import EditAnnotator
from tests.unit.cli.test_main import Cli


def test_reports_default_to_runs_with_a_timestamp(tmp_path: Path) -> None:
    path = _default_output(Cli(tmp_path).environment(), Dataset.JFLEG_DEV)
    assert path == tmp_path / DEFAULT_RUNS / "jfleg-dev-20261028T000000.json"


def test_the_real_annotator_loads_errant_lazily() -> None:
    annotator: EditAnnotator = load_edit_annotator()

    assert annotator.reference_edits(("a", "dog"), ("a", "dog")) == []
