from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pytest
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from langcheck.cli.tagging import (
    FAILED,
    TaggerTrainingError,
    Trained,
    TransformerRuntime,
    render_flag,
    tagger_app,
)
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.models import SlotScores
from langcheck.tagging.protocols import GapCommaScorer
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier
from tests.unit.cli.test_main import Cli

M2 = """S He go home .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0

S Fine .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
"""


class OnlySecondWord:
    def score(self, sentences: Sequence[Sequence[str]]) -> list[SlotScores]:
        return [
            SlotScores(
                tuple(0.9 if index == 1 else 0.0 for index in range(len(s))),
                (0.0,) * (len(s) + 1),
            )
            for s in sentences
        ]


@dataclass
class FakeRuntime:
    trained_on: list[int] = field(default_factory=list)
    commas: GapCommaScorer | None = None
    comma_runs: list[tuple[str, int, str]] = field(default_factory=list)

    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        base_model: str,
        output: Path,
        epochs: int,
        batch_size: int,
        precision: float,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.trained_on.append(len(sentences))
        log(f"{base_model} {epochs} {batch_size} {precision} {device} {output.name}")

    def load(self, path: Path, *, device: str) -> Trained:
        return Trained(OnlySecondWord(), 0.5, f"{path.name}-{device}", 0.5, self.commas)

    def train_commas(
        self,
        path: Path,
        sentences: Sequence[AnnotatedSentence],
        *,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.comma_runs.append((path.name, len(sentences), device))
        log("trained")


def run(cli: Cli, runtime: FakeRuntime, *argv: str) -> int:
    app = tagger_app(cli.environment(), runtime=runtime, guarded=lambda body: body())
    code = app(list(argv))
    assert isinstance(code, int)
    return code


def write(root: Path, dataset: Dataset) -> None:
    [path] = dataset_files(dataset, root / "data" / "eval")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(M2, encoding="utf-8")


def test_train_reads_both_training_sets(tmp_path: Path) -> None:
    write(tmp_path, Dataset.BEA19_TRAIN)
    write(tmp_path, Dataset.FCE_TRAIN)
    cli, runtime = Cli(tmp_path), FakeRuntime()

    assert run(cli, runtime, "train", "out", "--device", "cpu") == 0

    assert runtime.trained_on == [4]
    assert "roberta-large 3 16 0.5 cpu out" in cli.output


def test_train_asks_for_missing_sets_to_be_fetched(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "train", "out") == FAILED

    assert "evaluate fetch bea19-train" in cli.err[0]


def test_evaluate_prints_and_saves_the_report(tmp_path: Path) -> None:
    write(tmp_path, Dataset.CONLL14_TEST)
    cli = Cli(tmp_path)
    saved = tmp_path / "runs" / "report.json"

    code = run(cli, FakeRuntime(), "evaluate", "ckpt", "conll14-test", "--output", str(saved))

    assert code == 0
    assert "conll14-test: 2 sentences, ckpt-cuda" in cli.output
    assert saved.is_file()


def test_evaluate_asks_for_a_missing_set_to_be_fetched(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "evaluate", "ckpt", "conll14-test") == FAILED


def test_flag_prints_offsets_and_a_tier(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "flag", "ckpt", "He go home.") == 0

    assert cli.out == ["3-5\t0.900\terror\tgo"]


def test_flag_lowers_the_tier_with_a_higher_error_threshold(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "flag", "ckpt", "He go home.", "--error", "0.95") == 0

    assert cli.out == ["3-5\t0.900\tsuggestion\tgo"]


def test_flag_on_text_without_words_loads_nothing(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "flag", "missing", "  ") == 0
    assert cli.out == []


class GapBeforeLastWord:
    def score(self, sentences: Sequence[Sequence[str]]) -> list[SlotScores]:
        return [SlotScores((0.0,) * len(s), (0.0, 0.9, 0.0)) for s in sentences]


class AllCommas:
    def score(self, sentences: Sequence[Sequence[str]]) -> list[tuple[float, ...]]:
        return [(0.9,) * (len(s) + 1) for s in sentences]


@dataclass
class CommaRuntime(FakeRuntime):
    @override
    def load(self, path: Path, *, device: str) -> Trained:
        return Trained(GapBeforeLastWord(), 0.5, "m", 0.5, AllCommas())


def test_flag_can_hide_optional_commas(tmp_path: Path) -> None:
    shown = Cli(tmp_path)
    hidden = Cli(tmp_path)

    assert run(shown, CommaRuntime(), "flag", "ckpt", "Hi there") == 0
    assert run(hidden, CommaRuntime(), "flag", "ckpt", "Hi there", "--hide-commas") == 0

    assert shown.out == ["2-2\t0.900\toptional-comma\tHi^ there"]
    assert hidden.out == []


def test_a_flag_on_nothing_shows_where_it_is() -> None:
    gap = TextFlag(CharSpan(2, 2), 0.5, Tier.SUGGESTION, FlagKind.GENERAL)
    word = TextFlag(CharSpan(0, 2), 0.5, Tier.ERROR, FlagKind.GENERAL)

    assert render_flag("Go home", gap) == "2-2\t0.500\tsuggestion\tGo^ home"
    assert render_flag("Go", word) == "0-2\t0.500\terror\tGo"


def test_train_commas_reads_both_training_sets(tmp_path: Path) -> None:
    write(tmp_path, Dataset.BEA19_TRAIN)
    write(tmp_path, Dataset.FCE_TRAIN)
    cli, runtime = Cli(tmp_path), FakeRuntime()

    assert run(cli, runtime, "train-commas", "ckpt", "--device", "cpu") == 0

    assert runtime.comma_runs == [("ckpt", 4, "cpu")]


def test_train_commas_asks_for_missing_sets_to_be_fetched(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "train-commas", "ckpt") == FAILED


def test_tiers_prints_and_saves_the_report(tmp_path: Path) -> None:
    write(tmp_path, Dataset.CONLL14_TEST)
    cli = Cli(tmp_path)
    saved = tmp_path / "runs" / "tiers.json"

    code = run(cli, CommaRuntime(), "tiers", "ckpt", "conll14-test", "--output", str(saved))

    assert code == 0
    assert "conll14-test: 2 sentences" in cli.output
    assert saved.is_file()


def test_tiers_asks_for_a_missing_set_to_be_fetched(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "tiers", "ckpt", "conll14-test") == FAILED


def test_too_few_sentences_for_validation_are_refused(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentences = [AnnotatedSentence(("Fine", "."), ((),))]
    tiny_tokenizer.save_pretrained(tmp_path)
    config = RobertaConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", lambda _name: RobertaModel(config)
    )

    with pytest.raises(TaggerTrainingError, match="none for validation"):
        TransformerRuntime().train(
            sentences,
            base_model=str(tmp_path),
            output=tmp_path / "out",
            epochs=1,
            batch_size=1,
            precision=0.5,
            device="cpu",
            log=lambda _: None,
        )
