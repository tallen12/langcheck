from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langcheck.cli.proposing import FAILED, RecallRow, proposer_app, render_rows
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.proposal.protocols import Proposal, ProposalRequest, ProposalSource
from langcheck.tagging.models import SentenceReading, SlotScores
from langcheck.tagging.protocols import SentenceReader
from tests.unit.cli.test_main import Cli

M2 = """S He go home .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0

"""


class Source:
    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        return [[(Proposal("goes", -0.5),) for _ in request.spans] for request in requests]


class Reader:
    def read(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SentenceReading]:
        return [
            SentenceReading(
                SlotScores(
                    tuple(0.9 if word == "go" else 0.0 for word in s), (0.0,) * (len(s) + 1)
                ),
                None,
            )
            for s in sentences
        ]


@dataclass
class FakeRuntime:
    trained: list[int] = field(default_factory=list)

    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        tagger: Path,
        output: Path,
        epochs: int,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.trained.append(len(sentences))

    def load(
        self, tagger: Path, proposer: Path, *, device: str, width: int
    ) -> tuple[ProposalSource, SentenceReader, int]:
        return Source(), Reader(), 0


def run(cli: Cli, runtime: FakeRuntime, *argv: str) -> int:
    app = proposer_app(cli.environment(), runtime=runtime, guarded=lambda body: body())
    code = app(list(argv))
    assert isinstance(code, int)
    return code


def write_sets(root: Path, *datasets: Dataset) -> None:
    for dataset in datasets:
        [path] = dataset_files(dataset, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(M2, encoding="utf-8")


def test_training_reads_both_sets(tmp_path: Path) -> None:
    write_sets(tmp_path, Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
    runtime = FakeRuntime()

    code = run(Cli(tmp_path), runtime, "train", "out", "--tagger", "t", "--data", str(tmp_path))

    assert code == 0
    assert runtime.trained == [2]


def test_training_without_the_sets_fails(tmp_path: Path) -> None:
    code = run(Cli(tmp_path), FakeRuntime(), "train", "o", "--tagger", "t", "--data", str(tmp_path))

    assert code == FAILED


def test_measure_reports_recall_and_saves_it(tmp_path: Path) -> None:
    write_sets(tmp_path, Dataset.BEA19_DEV, Dataset.CONLL14_TEST)
    output = tmp_path / "rows.json"

    code = run(
        Cli(tmp_path),
        FakeRuntime(),
        "measure",
        "p",
        "--tagger",
        "t",
        "--data",
        str(tmp_path),
        "--output",
        str(output),
    )

    assert code == 0
    assert "conll14-test flags proposer" in output.read_text(encoding="utf-8")


def test_rows_render_each_cutoff() -> None:
    row = RecallRow(scoring="x", cases=1, recall={1: 0.5}, deletions_first=0.0)

    assert "@1 0.500" in render_rows([row])
