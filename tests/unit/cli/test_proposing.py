from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langcheck.cli.proposing import (
    FAILED,
    EpochScoring,
    Fresh,
    FromProposer,
    Measured,
    ProposerStart,
    RecallRow,
    epoch_score_line,
    proposer_app,
    render_rows,
    unfiltered_line,
)
from langcheck.correction.measure import CorrectionReport, Strategy, StrategyRow
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
from langcheck.proposal.protocols import Proposal, ProposalRequest, ProposalSource
from langcheck.tagging.models import SentenceReading, SlotScores
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import Adjacent, Falloff, TierPolicy
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
    false_alarms: list[TierPolicy] = field(default_factory=list)
    stages: list[tuple[ProposerStart, float, bool, bool]] = field(default_factory=list)
    tables: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    pairs: list[int] = field(default_factory=list)

    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        tagger: Path,
        output: Path,
        start: ProposerStart,
        epochs: int,
        learning_rate: float,
        real: bool,
        synthetic: int,
        corrected: bool,
        clean: Sequence[str],
        pairs: Sequence[tuple[AnnotatedSentence, str]],
        tables: Sequence[AnnotatedSentence],
        errors: Sequence[int],
        false_alarms: TierPolicy,
        scoring: EpochScoring | None,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.trained.append(len(sentences))
        self.synthetic.append(synthetic)
        self.scoring.append(scoring)
        self.false_alarms.append(false_alarms)
        self.stages.append((start, learning_rate, real, corrected))
        self.tables.append((len(tables), tuple(errors)))
        self.pairs.append(len(pairs))

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
    row = RecallRow(scoring="x", cases=1, recall={1: 0.5}, mrr=0.25, deletions_first=0.0)

    assert "@1 0.500" in render_rows([row])
    assert "MRR 0.250" in render_rows([row])


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
    assert [sentence.tokens for sentence in scoring.sentences] == [("He", "go", "home", ".")]


def test_training_also_scores_every_sentence_of_more_sets(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    extra = tmp_path / "held.m2"
    extra.write_text("\n".join([M2.strip() + "\n"] * 3), encoding="utf-8")
    runtime = FakeRuntime()
    common = ("train", "out", "--tagger", "t", "--data", str(tmp_path), "--scored", "1")

    assert run(Cli(tmp_path), runtime, *common, "--score-on", str(extra)) == 0

    [scoring] = runtime.scoring
    assert scoring is not None
    [(name, found)] = scoring.extra
    every = tuple((s, "") for s in read_m2(extra.read_text(encoding="utf-8")))
    assert len(every) == 3
    assert (name, found) == ("held", every)


def test_more_scoring_sets_need_scoring(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    cli, runtime = Cli(tmp_path), FakeRuntime()
    common = ("train", "out", "--tagger", "t", "--data", str(tmp_path), "--scored", "0")

    assert run(cli, runtime, *common, "--score-on", "x.m2") == FAILED

    assert runtime.trained == []
    assert "--score-on" in "".join(cli.err)


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


def test_epoch_score_line_shows_the_tagger_ranked_scores_then_ranking() -> None:
    recalls = {
        Strategy.TAGGER_RANKED: 0.25,
        Strategy.BEST_OF_3: 0.5,
        Strategy.BEST_OF_5: 0.625,
        Strategy.BEST_OFFERED: 0.75,
    }
    rows = [
        StrategyRow(
            strategy=strategy,
            tp=1,
            fp=1,
            fn=3,
            precision=0.5,
            recall=recalls.get(strategy, 0.0),
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
        mrr=0.4,
        strategies=rows,
        fates=[],
        order=[],
        records=[],
    )

    line = epoch_score_line(3, TierPolicy(suggestion=0.05), report)

    assert line == (
        "epoch 3: bea19-dev 2 sentences, flags from 0.05, tagger-ranked precision 0.500 "
        "recall 0.250 F0.5 0.4167 gain 1/4, MRR 0.400, recall in top 3 0.500 top 5 0.625 "
        "any 0.750"
    )


def test_training_scores_at_each_threshold_and_flags_from_false_alarms(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()

    code = run(
        Cli(tmp_path),
        runtime,
        "train",
        "--tagger",
        "t",
        "--false-alarms",
        "0.02",
        "--data",
        str(tmp_path),
        "--suggestion",
        "0.3",
        "--suggestion",
        "0.05",
        "out",
    )

    assert code == 0
    assert runtime.trained == [2]
    [scoring] = runtime.scoring
    assert scoring is not None
    assert [tiers.suggestion for tiers in scoring.tiers] == [0.3, 0.05]
    assert [policy.suggestion for policy in runtime.false_alarms] == [0.02]


def test_falloff_grows_training_and_scoring_flags_and_proposals_can_be_scored_alone(
    tmp_path: Path,
) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()

    code = run(
        Cli(tmp_path),
        runtime,
        "train",
        "out",
        "--tagger",
        "t",
        "--data",
        str(tmp_path),
        "--join",
        "0.1",
        "--rate",
        "3",
        "--measured",
        "proposals",
    )

    assert code == 0
    [scoring] = runtime.scoring
    assert scoring is not None
    assert scoring.measured is Measured.PROPOSALS
    assert [tiers.growth for tiers in scoring.tiers] == [Falloff(join=0.1, rate=3.0)]
    assert [policy.growth for policy in runtime.false_alarms] == [Falloff(join=0.1, rate=3.0)]


def test_flags_only_join_without_falloff_and_the_pipeline_is_scored_by_default(
    tmp_path: Path,
) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()

    run(Cli(tmp_path), runtime, "train", "out", "--tagger", "t", "--data", str(tmp_path))

    [scoring] = runtime.scoring
    assert scoring is not None
    assert scoring.measured is Measured.PIPELINE
    assert [tiers.growth for tiers in scoring.tiers] == [Adjacent()]


def test_training_defaults_to_the_user_and_false_alarm_thresholds(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()

    run(Cli(tmp_path), runtime, "train", "out", "--tagger", "t", "--data", str(tmp_path))

    [scoring] = runtime.scoring
    assert scoring is not None
    assert [tiers.suggestion for tiers in scoring.tiers] == [0.1]
    assert [policy.suggestion for policy in runtime.false_alarms] == [0.1]


def test_unfiltered_line_shows_the_mrr_and_recall_at_each_cutoff() -> None:
    row = RecallRow(
        scoring="proposer", cases=7, recall={1: 0.5, 16: 0.75}, mrr=0.6, deletions_first=0.0
    )

    line = unfiltered_line(2, TierPolicy(), row)

    assert line == (
        "epoch 2: unfiltered proposals at flags from 0.10, 7 spans with an error, MRR 0.600, "
        "recall @1 0.500 @16 0.750"
    )


def test_training_can_start_from_synthetic_data_alone_and_continue(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    runtime = FakeRuntime()
    common = ("train", "out", "--tagger", "t", "--scored", "0", "--data", str(tmp_path))

    first = ("--no-real", "--synthetic", "1", "--corrected")
    assert run(Cli(tmp_path), runtime, *common, *first) == 0
    then = ("--start-from", "first", "--learning-rate", "1e-4")
    assert run(Cli(tmp_path), runtime, *common, *then) == 0

    assert runtime.stages == [
        (Fresh(), 3e-4, False, True),
        (FromProposer(Path("first")), 1e-4, True, False),
    ]


def test_training_learns_a_second_set_of_tables_and_draws_error_counts(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    [path] = dataset_files(TRAINING[0], tmp_path)
    runtime = FakeRuntime()
    common = ("train", "out", "--tagger", "t", "--scored", "0", "--data", str(tmp_path))
    tables = ("--tables", str(path), "--errors", "0", "1", "2", "3")

    assert run(Cli(tmp_path), runtime, *common, "--synthetic", "1", *tables) == 0

    found = read_m2(path.read_text(encoding="utf-8"))
    assert runtime.tables == [(len(found), (0, 1, 2, 3))]


def test_training_can_take_pairs_alone(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    [path] = dataset_files(TRAINING[0], tmp_path)
    runtime = FakeRuntime()
    common = ("train", "out", "--tagger", "t", "--scored", "0", "--data", str(tmp_path))

    assert run(Cli(tmp_path), runtime, *common, "--no-real", "--pairs", str(path)) == 0

    assert runtime.pairs == [len(read_m2(path.read_text(encoding="utf-8")))]
    assert runtime.stages[0][2] is False


def test_training_refuses_a_negative_error_count(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    cli, runtime = Cli(tmp_path), FakeRuntime()
    common = ("train", "out", "--tagger", "t", "--scored", "0", "--data", str(tmp_path))

    assert run(cli, runtime, *common, "--errors=-1") == FAILED

    assert runtime.trained == []
    assert "--errors" in "".join(cli.err)


def test_training_without_real_or_synthetic_sentences_fails(tmp_path: Path) -> None:
    write_sets(tmp_path, *TRAINING)
    cli, runtime = Cli(tmp_path), FakeRuntime()

    code = run(cli, runtime, "train", "out", "--tagger", "t", "--no-real", "--data", str(tmp_path))

    assert code == FAILED
    assert runtime.trained == []
    assert "--synthetic, --clean or --pairs" in "".join(cli.err)
