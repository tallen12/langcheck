from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pytest
import torch
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from langcheck.cli.tagging import (
    FAILED,
    Activations,
    Continued,
    Pretrained,
    Start,
    Trained,
    TransformerRuntime,
    read_pair_file,
    render_flag,
    tagger_app,
)
from langcheck.evaluation.context_pairs import EvaluationDataError, Origin, write_pairs
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit, read_m2
from langcheck.synthesis.training_mix import TrainingMix, TrainingMixError
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.checkpoint import Checkpoint, CheckpointInfo
from langcheck.tagging.model import SlotTagger
from langcheck.tagging.models import SlotScores
from langcheck.tagging.protocols import GapCommaScorer
from langcheck.tagging.selection import ByFlags, BySlots, EpochChoice
from langcheck.tagging.tiers import Falloff, FlagKind, TextFlag, Tier, TierPolicy
from langcheck.tagging.training import InContext
from tests.unit.cli.test_main import Cli

M2 = """S He go home .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0

S Fine .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
"""


class OnlySecondWord:
    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        return [
            SlotScores(
                tuple(0.9 if index == 1 else 0.0 for index in range(len(s))),
                (0.0,) * (len(s) + 1),
            )
            for s in sentences
        ]


class StopTraining(Exception):
    """Ends a runtime's training once its sentences are assembled."""


@dataclass
class FakeRuntime:
    trained_on: list[int] = field(default_factory=list)
    context: list[int] = field(default_factory=list)
    corrected: list[bool] = field(default_factory=list)
    pairs: list[int] = field(default_factory=list)
    tables: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    validate_on: list[dict[str, int]] = field(default_factory=list)
    choices: list[EpochChoice] = field(default_factory=list)
    starts: list[tuple[Start, float, float, Activations]] = field(default_factory=list)
    commas: GapCommaScorer | None = None
    comma_runs: list[tuple[str, int, str]] = field(default_factory=list)
    comma_mixes: list[TrainingMix] = field(default_factory=list)

    def train(
        self,
        *,
        start: Start,
        output: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        error_weight: float,
        activations: Activations,
        precision: float,
        choice: EpochChoice,
        context_sentences: int,
        mix: TrainingMix,
        validate_on: Mapping[str, Sequence[tuple[AnnotatedSentence, str]]],
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.trained_on.append(len(mix.clean))
        self.choices.append(choice)
        self.context.append(context_sentences)
        self.corrected.append(mix.corrected)
        self.pairs.append(len(mix.pairs))
        self.tables.append((len(mix.tables), mix.errors))
        self.validate_on.append({name: len(found) for name, found in validate_on.items()})
        self.starts.append((start, learning_rate, error_weight, activations))
        log(f"{epochs} {batch_size} {precision} {device} {output.name}")

    def load(self, path: Path, *, device: str) -> Trained:
        return Trained(OnlySecondWord(), 0.5, f"{path.name}-{device}", 0.5, self.commas)

    def train_commas(
        self,
        path: Path,
        *,
        mix: TrainingMix,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        self.comma_runs.append((path.name, len(mix.pairs), device))
        self.comma_mixes.append(mix)
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


def prose(root: Path) -> str:
    """A JSONL file of two edited texts, as --clean reads it."""
    path = root / "clean.jsonl"
    path.write_text('{"text": "All is well."}\n{"text": "It was late."}\n', encoding="utf-8")
    return str(path)


def test_train_corrupts_clean_prose(tmp_path: Path) -> None:
    cli, runtime = Cli(tmp_path), FakeRuntime()

    assert run(cli, runtime, "train", "out", "--clean", prose(tmp_path), "--device", "cpu") == 0

    assert runtime.trained_on == [2]
    assert runtime.starts == [(Pretrained("roberta-large"), 2e-5, 1.0, Activations.KEEP)]
    assert "3 16 0.5 cpu out" in cli.output


def test_train_can_continue_a_saved_tagger_with_weighted_errors_and_less_memory(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()

    code = run(
        Cli(tmp_path),
        runtime,
        "train",
        "out",
        "--clean",
        prose(tmp_path),
        "--start-from",
        "runs/synthetic",
        "--learning-rate",
        "1e-5",
        "--error-weight",
        "3",
        "--activations",
        "recompute",
    )

    assert code == 0
    assert runtime.starts == [(Continued(Path("runs/synthetic")), 1e-5, 3.0, Activations.RECOMPUTE)]


def test_train_keeps_the_best_slot_recall_unless_judged_by_flags(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    clean = ("--clean", prose(tmp_path))

    run(Cli(tmp_path), runtime, "train", "out", *clean)
    run(Cli(tmp_path), runtime, "train", "out", *clean, "--flags-from", "0.3", "--join", "0.1")

    assert runtime.choices == [
        BySlots(),
        ByFlags(TierPolicy(suggestion=0.3, growth=Falloff(join=0.1, rate=2.0))),
    ]


def test_train_can_read_preceding_sentences(tmp_path: Path) -> None:
    runtime = FakeRuntime()

    assert (
        run(Cli(tmp_path), runtime, "train", "out", "--clean", prose(tmp_path), "--context", "2")
        == 0
    )

    assert runtime.context == [2]


def test_train_can_add_each_sentences_correction(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    clean = ("--clean", prose(tmp_path))

    assert run(Cli(tmp_path), runtime, "train", "out", *clean) == 0
    assert run(Cli(tmp_path), runtime, "train", "out", *clean, "--corrected") == 0

    assert runtime.corrected == [False, True]


def test_train_needs_prose_or_pairs(tmp_path: Path) -> None:
    cli, runtime = Cli(tmp_path), FakeRuntime()

    assert run(cli, runtime, "train", "out") == FAILED

    assert runtime.trained_on == []
    assert "--clean prose or --pairs" in cli.err[0]


def test_train_adds_the_sentences_of_each_pairs_file(tmp_path: Path) -> None:
    (tmp_path / "a.m2").write_text(M2, encoding="utf-8")
    (tmp_path / "b.m2").write_text(M2, encoding="utf-8")
    runtime = FakeRuntime()

    a, b = str(tmp_path / "a.m2"), str(tmp_path / "b.m2")

    code = run(Cli(tmp_path), runtime, "train", "out", "--pairs", a, "--pairs", b)

    assert code == 0
    assert runtime.pairs == [2 * len(read_m2(M2))]
    assert runtime.trained_on == [0]


def test_jsonl_pairs_keep_their_context_and_can_be_validated_on(tmp_path: Path) -> None:
    pair = AnnotatedSentence(("He", "go", "."), ((GoldEdit(1, 2, ("goes",), "R:VERB:SVA"),),))
    (tmp_path / "writers.jsonl").write_text(write_pairs([(pair, "It was late.")]), encoding="utf-8")
    runtime = FakeRuntime()
    path = str(tmp_path / "writers.jsonl")

    assert read_pair_file(tmp_path / "writers.jsonl") == [(pair, "It was late.")]
    assert run(Cli(tmp_path), runtime, "train", "out", "--pairs", path, "--validate-on", path) == 0

    assert runtime.pairs == [1]
    assert runtime.validate_on == [{"writers": 1}]


def test_train_learns_a_second_set_of_tables_and_draws_error_counts(tmp_path: Path) -> None:
    (tmp_path / "t.m2").write_text(M2, encoding="utf-8")
    runtime = FakeRuntime()
    clean = ("--clean", prose(tmp_path))
    tables = ("--tables", str(tmp_path / "t.m2"), "--errors", "0", "1", "2", "3")

    assert run(Cli(tmp_path), runtime, "train", "out", *clean, *tables) == 0
    assert run(Cli(tmp_path), runtime, "train", "out", *clean) == 0

    assert runtime.tables == [(len(read_m2(M2)), (0, 1, 2, 3)), (0, ())]


def test_train_refuses_a_negative_error_count(tmp_path: Path) -> None:
    cli, runtime = Cli(tmp_path), FakeRuntime()

    assert run(cli, runtime, "train", "out", "--clean", prose(tmp_path), "--errors=-1") == FAILED

    assert runtime.tables == []
    assert "--errors" in cli.err[0]


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


def test_flag_covers_every_sentence_of_the_text(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "flag", "ckpt", "He go home. We go out.") == 0

    assert cli.out == ["3-5\t0.900\terror\tgo", "15-17\t0.900\terror\tgo"]


def test_flag_lowers_the_tier_with_a_higher_error_threshold(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "flag", "ckpt", "He go home.", "--error", "0.95") == 0

    assert cli.out == ["3-5\t0.900\tsuggestion\tgo"]


def test_flag_on_text_without_words_loads_nothing(tmp_path: Path) -> None:
    cli = Cli(tmp_path)

    assert run(cli, FakeRuntime(), "flag", "missing", "  ") == 0
    assert cli.out == []


class GapBeforeLastWord:
    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        return [SlotScores((0.0,) * len(s), (0.0, 0.9, 0.0)) for s in sentences]


class AllCommas:
    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[tuple[float, ...]]:
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


def test_train_commas_reads_the_training_mix(tmp_path: Path) -> None:
    (tmp_path / "p.m2").write_text(M2, encoding="utf-8")
    cli, runtime = Cli(tmp_path), FakeRuntime()
    pairs = str(tmp_path / "p.m2")

    assert run(cli, runtime, "train-commas", "ckpt", "--pairs", pairs, "--device", "cpu") == 0

    assert runtime.comma_runs == [("ckpt", 2, "cpu")]


def test_train_commas_takes_the_same_training_mix_as_the_tagger(tmp_path: Path) -> None:
    (tmp_path / "p.m2").write_text(M2, encoding="utf-8")
    runtime = FakeRuntime()
    mix = ("--clean", prose(tmp_path), "--pairs", str(tmp_path / "p.m2"), "--errors", "0", "1")

    assert run(Cli(tmp_path), runtime, "train-commas", "ckpt", *mix) == 0

    [found] = runtime.comma_mixes
    assert (found.clean, found.errors) == (("All is well.", "It was late."), (0, 1))
    assert len(found.pairs) == len(read_m2(M2))


def test_train_commas_needs_prose_or_pairs(tmp_path: Path) -> None:
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


def test_the_encoder_loads_in_float32_and_too_few_sentences_are_refused(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = ((AnnotatedSentence(("Fine", "."), ((),)), ""),)
    tiny_tokenizer.save_pretrained(tmp_path)
    config = RobertaConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
    )
    asked: list[torch.dtype] = []

    def pretrained(_name: str, *, dtype: torch.dtype) -> RobertaModel:
        asked.append(dtype)
        return RobertaModel(config)

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", pretrained)

    with pytest.raises(TrainingMixError, match="none for validation"):
        TransformerRuntime().train(
            start=Pretrained(str(tmp_path)),
            output=tmp_path / "out",
            epochs=1,
            batch_size=1,
            learning_rate=2e-5,
            error_weight=1.0,
            activations=Activations.KEEP,
            precision=0.5,
            choice=BySlots(),
            context_sentences=0,
            mix=TrainingMix(pairs=pairs),
            validate_on={},
            device="cpu",
            log=lambda _: None,
        )

    assert asked == [torch.float32]


def test_continuing_loads_the_saved_tagger_and_can_recompute_activations(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RobertaConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
    )
    info = CheckpointInfo(
        base_model="roberta-large",
        threshold=0.13,
        precision_target=0.5,
        max_length=256,
        train_sentences=10,
        validation_sentences=1,
        epoch=2,
    )
    loaded: list[Path] = []
    tagger = SlotTagger(RobertaModel(config))

    def load(directory: Path, *, device: torch.device) -> Checkpoint:
        loaded.append(directory)
        return Checkpoint(tagger.to(device), tiny_tokenizer, info)

    monkeypatch.setattr("langcheck.tagging.checkpoint.load_checkpoint", load)
    lines: list[str] = []

    with pytest.raises(TrainingMixError, match="none for validation"):
        TransformerRuntime().train(
            start=Continued(tmp_path / "synthetic"),
            output=tmp_path / "out",
            epochs=1,
            batch_size=1,
            learning_rate=1e-5,
            error_weight=1.0,
            activations=Activations.RECOMPUTE,
            precision=0.5,
            choice=BySlots(),
            context_sentences=0,
            mix=TrainingMix(pairs=((AnnotatedSentence(("Fine", "."), ((),)), ""),)),
            validate_on={},
            device="cpu",
            log=lines.append,
        )

    assert loaded == [tmp_path / "synthetic"]
    assert tagger.encoder.is_gradient_checkpointing
    assert lines == [f"continuing from {tmp_path / 'synthetic'} (epoch 2)"]


def test_pairs_keep_context_and_validation_mixes_named_sets(
    tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    fine = (AnnotatedSentence(("Fine", "."), ((),)), "")
    pair = AnnotatedSentence(("He", "go", "."), ((GoldEdit(1, 2, ("goes",), "R:VERB:SVA"),),))
    tiny_tokenizer.save_pretrained(tmp_path)
    config = RobertaConfig(
        vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, intermediate_size=8
    )
    seen: list[tuple[list[InContext], list[InContext]]] = []

    def pretrained(_name: str, *, dtype: torch.dtype) -> RobertaModel:
        return RobertaModel(config)

    def stop(
        _tagger: object,
        _tokenizer: object,
        training: list[InContext],
        validation: list[InContext],
        **_: object,
    ) -> None:
        seen.append((training, validation))
        raise StopTraining

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", pretrained)
    monkeypatch.setattr("langcheck.tagging.training.train", stop)

    with pytest.raises(StopTraining):
        TransformerRuntime().train(
            start=Pretrained(str(tmp_path)),
            output=tmp_path / "out",
            epochs=1,
            batch_size=1,
            learning_rate=2e-5,
            error_weight=1.0,
            activations=Activations.KEEP,
            precision=0.5,
            choice=BySlots(),
            context_sentences=0,
            mix=TrainingMix(corrected=True, pairs=((pair, "It was late."), *[fine] * 2000)),
            validate_on={"writers": [(pair, "It was late.")]},
            device="cpu",
            log=lambda _: None,
        )

    [(training, validation)] = seen
    assert training[0] == InContext(pair, "It was late.")
    assert training[-1] == InContext(AnnotatedSentence(("He", "goes", "."), ((),)), "It was late.")
    assert len(training) == 1901 + 1
    assert len(validation) == 100 + 1
    assert {item.source for item in validation[:-1]} == {"held-out"}
    assert validation[-1] == InContext(pair, "It was late.", "writers")


def test_training_refuses_pairs_made_from_an_evaluation_set(tmp_path: Path) -> None:
    pair = AnnotatedSentence(("He", "go", "."), ((GoldEdit(1, 2, ("goes",), "R:VERB:SVA"),),))
    marked = tmp_path / "eval.jsonl"
    marked.write_text(write_pairs([(pair, "")], origin=Origin.EVALUATION), encoding="utf-8")
    runtime = FakeRuntime()

    with pytest.raises(EvaluationDataError, match="evaluation set"):
        run(Cli(tmp_path), runtime, "train", "out", "--pairs", str(marked))
    with pytest.raises(EvaluationDataError):
        run(
            Cli(tmp_path),
            runtime,
            "train",
            "out",
            "--clean",
            prose(tmp_path),
            "--tables",
            str(marked),
        )
    assert (
        run(
            Cli(tmp_path),
            runtime,
            "train",
            "out",
            "--clean",
            prose(tmp_path),
            "--validate-on",
            str(marked),
        )
        == 0
    )

    assert runtime.validate_on == [{"eval": 1}]
