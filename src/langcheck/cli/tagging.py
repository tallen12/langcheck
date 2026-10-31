"""``langcheck tagger``: train the per-token error tagger, evaluate it and flag text."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.examples import load_examples
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
from langcheck.tagging.alignment import GapFlag, TextFlag, WordFlag, flag_text
from langcheck.tagging.metrics import ScoredSentence
from langcheck.tagging.protocols import SlotScorer
from langcheck.tagging.report import build_report, render_report

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
DEFAULT_TRAINING_SETS = (Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
DEFAULT_MODEL = "roberta-large"

type Guard = Callable[[Callable[[], int]], int]


class TaggerTrainingError(Exception):
    """There are too few sentences to train on and choose a threshold with."""


@dataclass(frozen=True, slots=True)
class Trained:
    """A loaded tagger and what evaluating it needs to know."""

    scorer: SlotScorer
    threshold: float
    base_model: str
    precision_target: float


class TaggerRuntime(Protocol):
    """Training and loading, which need PyTorch, kept away from the commands."""

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
        """Fine-tune ``base_model`` on ``sentences`` and save it to ``output``."""
        ...

    def load(self, path: Path, *, device: str) -> Trained:
        """The tagger saved at ``path``."""
        ...


class TransformerRuntime:
    """The real runtime, importing PyTorch only when a command needs it."""

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
        """Fine-tune ``base_model`` on ``sentences`` and save it to ``output``."""
        import torch
        from transformers import AutoModel

        from langcheck.tagging.checkpoint import CheckpointInfo, save_checkpoint
        from langcheck.tagging.encoding import load_tokenizer
        from langcheck.tagging.model import SlotTagger
        from langcheck.tagging.training import TrainingOptions, split_holdout, train

        options = TrainingOptions(epochs=epochs, batch_size=batch_size, precision=precision)
        target = torch.device(device)
        tokenizer = load_tokenizer(base_model)
        tagger = SlotTagger(AutoModel.from_pretrained(base_model)).to(target)
        training, validation = split_holdout(sentences)
        if not validation:
            msg = f"{len(sentences)} sentences leave none for validation; use more"
            raise TaggerTrainingError(msg)
        log(f"{len(training)} training sentences, {len(validation)} validation sentences")
        result = train(
            tagger, tokenizer, training, validation, device=target, options=options, log=log
        )
        save_checkpoint(
            output,
            tagger,
            tokenizer,
            CheckpointInfo(
                base_model=base_model,
                threshold=result.validation.threshold,
                precision_target=precision,
                max_length=options.max_length,
                train_sentences=len(training),
                validation_sentences=len(validation),
                epoch=result.epoch,
            ),
        )

    def load(self, path: Path, *, device: str) -> Trained:
        """The tagger saved at ``path``."""
        import torch

        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.inference import TaggerScorer

        target = torch.device(device)
        loaded = load_checkpoint(path, device=target)
        return Trained(
            TaggerScorer(
                loaded.tagger, loaded.tokenizer, device=target, max_length=loaded.info.max_length
            ),
            loaded.info.threshold,
            loaded.info.base_model,
            loaded.info.precision_target,
        )


def render_flag(text: str, flag: TextFlag) -> str:
    """One flag as its character offsets and what it is on."""
    match flag:
        case WordFlag(span=span, score=score):
            return f"{span.start}-{span.end}\t{score:.3f}\tword\t{text[span.start : span.end]}"
        case GapFlag(offset=offset, score=score):
            before = text[max(0, offset - 12) : offset]
            return f"{offset}-{offset}\t{score:.3f}\tgap\t{before}^{text[offset : offset + 12]}"


def tagger_app(env: CliEnvironment, *, runtime: TaggerRuntime, guarded: Guard) -> App:
    """The ``tagger`` commands."""
    app = App(
        name="tagger",
        help="Train and evaluate the per-token grammatical error tagger.",
        result_action="return_value",
    )
    type DeviceOption = Annotated[str, Parameter(help="PyTorch device, such as cpu or cuda.")]

    def data_dir(data: Path | None) -> Path:
        return data if data is not None else env.cwd / DEFAULT_DATA

    @app.command(name="train")
    def train_command(
        output: Path,
        /,
        *,
        model: Annotated[str, Parameter(help="Pretrained encoder to fine-tune.")] = DEFAULT_MODEL,
        epochs: int = 3,
        batch_size: int = 16,
        precision: Annotated[float, Parameter(help="Precision to choose a threshold at.")] = 0.5,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        device: DeviceOption = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Fine-tune on the BEA-2019 W&I+LOCNESS and FCE training sets; save to OUTPUT."""

        def run() -> int:
            sentences: list[AnnotatedSentence] = []
            for dataset in DEFAULT_TRAINING_SETS:
                [path] = dataset_files(dataset, data_dir(data))
                if not path.is_file():
                    env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                    return FAILED
                sentences += read_m2(path.read_text(encoding="utf-8"))
            runtime.train(
                sentences[:limit],
                base_model=model,
                output=output,
                epochs=epochs,
                batch_size=batch_size,
                precision=precision,
                device=device,
                log=env.out,
            )
            env.out(f"saved: {output}")
            return CLEAN

        return guarded(run)

    @app.command(name="evaluate")
    def evaluate_command(
        checkpoint: Path,
        dataset: Dataset,
        /,
        *,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        output: Annotated[Path | None, Parameter(help="Also save the report here.")] = None,
        device: DeviceOption = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Score a trained tagger's detection on DATASET, per kind of error."""

        def run() -> int:
            files = dataset_files(dataset, data_dir(data))
            if not all(path.is_file() for path in files):
                env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                return FAILED
            trained = runtime.load(checkpoint, device=device)
            examples = load_examples(dataset, files, env.edit_annotators(), limit=limit)
            scores = trained.scorer.score([example.source for example in examples])
            report = build_report(
                [
                    ScoredSentence(score, example.annotators)
                    for score, example in zip(scores, examples, strict=True)
                ],
                dataset=dataset.value,
                base_model=trained.base_model,
                threshold=trained.threshold,
                precision_target=trained.precision_target,
            )
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            env.out(render_report(report))
            return CLEAN

        return guarded(run)

    @app.command(name="flag")
    def flag_command(
        checkpoint: Path,
        text: str,
        /,
        *,
        threshold: Annotated[float | None, Parameter(help="Default: validation's.")] = None,
        device: DeviceOption = "cuda",
    ) -> int:
        """Print TEXT's flagged slots as character offsets, score, slot type and text."""

        def run() -> int:
            trained = runtime.load(checkpoint, device=device)
            limit = threshold if threshold is not None else trained.threshold
            for flag in flag_text(text, trained.scorer, threshold=limit):
                env.out(render_flag(text, flag))
            return CLEAN

        return guarded(run)

    return app
