from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langcheck.cli.proposing import (
    FAILED,
    EpochScoring,
    RecallRow,
    epoch_score_line,
    proposer_app,
    render_rows,
)
from langcheck.correction.measure import CorrectionReport, Strategy, StrategyRow
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
    synthetic: list[int] = field(default_factory=list)
    scoring: list[EpochScoring | None] = field(default_factory=list)

    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        tagger: Path,
        output: Path,
        epochs: int,
        synthetic: int,
        clean: Sequence[str],
        scoring: EpochScoring | None,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.trained.append(len(sentences))
        self.synthetic.append(synthetic)
        self.scoring.append(scoring)

    def load(
        self, tagger: Path, proposer: Path, *, device: str, width: int
    ) -> tuple[ProposalSource, SentenceReader, int]:
        return Source(), Reader(), 0


TRAINING = (Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN, Dataset.BEA19_DEV)


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
    write_sets(tmp_path, *TRAINING)
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


def test_training_passes_on_synthetic_copies(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()

    code = run(
        Cli(tmp_path),
        runtime,
        "train",
        "out",
        "--tagger",
        "t",
        "--synthetic",
        "2",
        "--data",
        str(tmp_path),
    )

    assert code == 0
    assert runtime.synthetic == [2]


def test_training_scores_on_bea19_dev_sentences(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()

    code = run(Cli(tmp_path), runtime, "train", "out", "--tagger", "t", "--data", str(tmp_path))

    assert code == 0
    [scoring] = runtime.scoring
    assert scoring is not None
    assert scoring.dataset == "bea19-dev"
    assert [example.source for example in scoring.examples] == [("He", "go", "home", ".")]


def test_training_without_scoring_needs_no_dev_set(tmp_path: Path) -> None:
    write_sets(tmp_path, Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
    runtime = FakeRuntime()

    code = run(
        Cli(tmp_path),
        runtime,
        "train",
        "out",
        "--tagger",
        "t",
        "--scored",
        "0",
        "--data",
        str(tmp_path),
    )

    assert code == 0
    assert runtime.scoring == [None]


def test_training_without_the_dev_set_fails(tmp_path: Path) -> None:
    write_sets(tmp_path, Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
    cli = Cli(tmp_path)

    code = run(cli, FakeRuntime(), "train", "out", "--tagger", "t", "--data", str(tmp_path))

    assert code == FAILED
    assert "evaluate fetch bea19-dev" in "".join(cli.err)


def test_epoch_score_line_shows_the_tagger_ranked_scores() -> None:
    rows = [
        StrategyRow(
            strategy=strategy,
            tp=1,
            fp=1,
            fn=3,
            precision=0.5,
            recall=0.25 if strategy is Strategy.TAGGER_RANKED else 0.0,
            f05=0.4167,
            needed=4,
            gain=1,
            share=0.25,
        )
        for strategy in Strategy
    ]
    report = CorrectionReport(
        dataset="bea19-dev",
        sentences=2,
        flags=2,
        unavailable=0,
        no_corrections=0,
        strategies=rows,
        fates=[],
        order=[],
        records=[],
    )

    line = epoch_score_line(3, report)

    assert line == (
        "epoch 3: bea19-dev 2 sentences, tagger-ranked precision 0.500 recall 0.250 "
        "F0.5 0.4167 gain 1/4"
    )
