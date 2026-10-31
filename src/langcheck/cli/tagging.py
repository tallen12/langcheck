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
from langcheck.tagging.alignment import preceding_text
from langcheck.tagging.document import flag_text
from langcheck.tagging.metrics import ScoredSentence
from langcheck.tagging.protocols import GapCommaScorer, SlotScorer
from langcheck.tagging.report import build_report, render_report
from langcheck.tagging.tier_report import TierSentence, measure_tiers, render_tiers
from langcheck.tagging.tiers import FlagKind, TextFlag, TierPolicy

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
DEFAULT_TRAINING_SETS = (Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
DEFAULT_MODEL = "roberta-large"
MAX_VRAM_SHARE = 0.8

DEFAULT_POLICY = TierPolicy()

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
    commas: GapCommaScorer | None = None
    context_sentences: int = 0
    """``None`` until ``langcheck tagger train-commas`` has run on the checkpoint."""


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
        context_sentences: int,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Fine-tune ``base_model`` on ``sentences`` and save it to ``output``."""
        ...

    def load(self, path: Path, *, device: str) -> Trained:
        """The tagger saved at ``path``."""
        ...

    def train_commas(
        self,
        path: Path,
        sentences: Sequence[AnnotatedSentence],
        *,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Add a comma head, fitted on ``sentences``, to the checkpoint at ``path``."""
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
        context_sentences: int,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Fine-tune ``base_model`` on ``sentences`` and save it to ``output``."""
        import torch
        from transformers import AutoModel

        from langcheck.tagging.checkpoint import CheckpointInfo, save_checkpoint
        from langcheck.tagging.encoding import load_tokenizer
        from langcheck.tagging.model import SlotTagger
        from langcheck.tagging.training import InContext, TrainingOptions, split_holdout, train

        options = TrainingOptions(epochs=epochs, batch_size=batch_size, precision=precision)
        target = torch.device(device)
        torch_limit(target)
        tokenizer = load_tokenizer(base_model)
        tagger = SlotTagger(AutoModel.from_pretrained(base_model)).to(target)
        contexts = preceding_text([s.tokens for s in sentences], context_sentences)
        items = [InContext(s, c) for s, c in zip(sentences, contexts, strict=True)]
        training, validation = split_holdout(items)
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
                context_sentences=context_sentences,
            ),
        )

    def load(self, path: Path, *, device: str) -> Trained:
        """The tagger saved at ``path``."""
        import torch

        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.comma import CommaScorer, has_comma_head, load_comma_head
        from langcheck.tagging.inference import TaggerScorer

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(path, device=target)
        length = loaded.info.max_length
        commas = None
        if has_comma_head(path):
            head = load_comma_head(path, loaded.tagger, device=target)
            commas = CommaScorer(
                loaded.tagger, head, loaded.tokenizer, device=target, max_length=length
            )
        return Trained(
            TaggerScorer(loaded.tagger, loaded.tokenizer, device=target, max_length=length),
            loaded.info.threshold,
            loaded.info.base_model,
            loaded.info.precision_target,
            commas,
            loaded.info.context_sentences,
        )

    def train_commas(
        self,
        path: Path,
        sentences: Sequence[AnnotatedSentence],
        *,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Add a comma head, fitted on ``sentences``, to the checkpoint at ``path``."""
        import torch

        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.comma import save_comma_head, train_comma_head
        from langcheck.tagging.training import InContext, split_holdout

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(path, device=target)
        contexts = preceding_text([s.tokens for s in sentences], loaded.info.context_sentences)
        items = [InContext(s, c) for s, c in zip(sentences, contexts, strict=True)]
        training, _ = split_holdout(items)
        head = train_comma_head(
            loaded.tagger,
            loaded.tokenizer,
            training,
            device=target,
            max_length=loaded.info.max_length,
            log=log,
        )
        save_comma_head(path, head)


def torch_limit(device: object) -> None:
    """Past VRAM the driver spills into system RAM, which can take the desktop down."""
    import torch

    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(MAX_VRAM_SHARE)


def render_flag(text: str, flag: TextFlag) -> str:
    """One flag as its character offsets, score, class and what it is on."""
    span = flag.span
    name = flag.kind.value if flag.kind is FlagKind.OPTIONAL_COMMA else flag.tier.value
    if span.start == span.end:
        before = text[max(0, span.start - 12) : span.start]
        shown = f"{before}^{text[span.start : span.start + 12]}"
    else:
        shown = text[span.start : span.end]
    return f"{span.start}-{span.end}\t{flag.score:.3f}\t{name}\t{shown}"


def tagger_app(env: CliEnvironment, *, runtime: TaggerRuntime, guarded: Guard) -> App:
    """The ``tagger`` commands."""
    app = App(
        name="tagger",
        help="Train and evaluate the per-token grammatical error tagger.",
        result_action="return_value",
    )
    type DeviceOption = Annotated[str, Parameter(help="PyTorch device, such as cpu or cuda.")]
    type ErrorOption = Annotated[float, Parameter(help="Score where the error tier begins.")]
    type SuggestionOption = Annotated[float, Parameter(help="Lowest score shown at all.")]

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
        context: Annotated[int, Parameter(help="Preceding sentences to read.")] = 0,
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
                context_sentences=context,
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
            tokens = [example.source for example in examples]
            scores = trained.scorer.score(
                tokens, contexts=preceding_text(tokens, trained.context_sentences)
            )
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

    @app.command(name="train-commas")
    def train_commas_command(
        checkpoint: Path,
        /,
        *,
        device: DeviceOption = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Add a head to CHECKPOINT that tells missing commas from other missing words."""

        def run() -> int:
            sentences: list[AnnotatedSentence] = []
            for dataset in DEFAULT_TRAINING_SETS:
                [path] = dataset_files(dataset, data_dir(data))
                if not path.is_file():
                    env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                    return FAILED
                sentences += read_m2(path.read_text(encoding="utf-8"))
            runtime.train_commas(checkpoint, sentences, device=device, log=env.out)
            env.out(f"saved: {checkpoint}")
            return CLEAN

        return guarded(run)

    @app.command(name="tiers")
    def tiers_command(
        checkpoint: Path,
        dataset: Dataset,
        /,
        *,
        error: ErrorOption = DEFAULT_POLICY.error,
        suggestion: SuggestionOption = DEFAULT_POLICY.suggestion,
        output: Annotated[Path | None, Parameter(help="Also save the report here.")] = None,
        device: DeviceOption = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Score the flags a reader would see on DATASET, tier by tier."""

        def run() -> int:
            files = dataset_files(dataset, data_dir(data))
            if not all(path.is_file() for path in files):
                env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                return FAILED
            trained = runtime.load(checkpoint, device=device)
            examples = load_examples(dataset, files, env.edit_annotators())
            tokens = [example.source for example in examples]
            contexts = preceding_text(tokens, trained.context_sentences)
            scores = trained.scorer.score(tokens, contexts=contexts)
            commas = (
                trained.commas.score(tokens, contexts=contexts)
                if trained.commas is not None
                else None
            )
            report = measure_tiers(
                [
                    TierSentence(
                        score, commas[index] if commas is not None else None, example.annotators
                    )
                    for index, (score, example) in enumerate(zip(scores, examples, strict=True))
                ],
                dataset=dataset.value,
                policy=TierPolicy(error=error, suggestion=suggestion),
            )
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            env.out(render_tiers(report))
            return CLEAN

        return guarded(run)

    @app.command(name="flag")
    def flag_command(
        checkpoint: Path,
        text: str,
        /,
        *,
        error: ErrorOption = DEFAULT_POLICY.error,
        suggestion: SuggestionOption = DEFAULT_POLICY.suggestion,
        hide_commas: Annotated[
            bool, Parameter(negative=(), help="Leave out optional commas.")
        ] = False,
        device: DeviceOption = "cuda",
    ) -> int:
        """Print the flags of TEXT, one or more sentences, as character offsets, score, class
        and text.
        """

        def run() -> int:
            if not text.strip():
                return CLEAN
            trained = runtime.load(checkpoint, device=device)
            flags = flag_text(
                text,
                trained.scorer,
                trained.commas,
                policy=TierPolicy(error=error, suggestion=suggestion),
                context_sentences=trained.context_sentences,
            )
            for flag in flags:
                if hide_commas and flag.kind is FlagKind.OPTIONAL_COMMA:
                    continue
                env.out(render_flag(text, flag))
            return CLEAN

        return guarded(run)

    return app
