"""``langcheck tagger``: train the per-token error tagger, evaluate it and flag text."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol, assert_never

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.evaluation.context_pairs import (
    EvaluationDataError,
    Origin,
    pairs_origin,
    read_pairs,
)
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.examples import load_examples
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
from langcheck.synthesis.clean_text import read_texts
from langcheck.synthesis.training_mix import TrainingMix
from langcheck.tagging.alignment import preceding_text
from langcheck.tagging.document import flag_text
from langcheck.tagging.loaded import Trained, load_trained, torch_limit
from langcheck.tagging.metrics import ScoredSentence
from langcheck.tagging.report import build_report, render_report
from langcheck.tagging.selection import ByFlags, BySlots, EpochChoice
from langcheck.tagging.tier_report import TierSentence, measure_tiers, render_tiers
from langcheck.tagging.tiers import Adjacent, Falloff, FlagKind, Growth, TextFlag, TierPolicy

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
DEFAULT_MODEL = "roberta-large"

DEFAULT_POLICY = TierPolicy()
HELD_SOURCE = "held-out"
"""How the held-out share of the training mix is named among the validation sets."""
PAIRS_SUFFIX = ".jsonl"

type Guard = Callable[[Callable[[], int]], int]


def read_pair_file(path: Path) -> list[tuple[AnnotatedSentence, str]]:
    """The sentences of ``path`` with the text before each: JSONL pairs carry their
    context, M2 sentences stand alone.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == PAIRS_SUFFIX:
        return read_pairs(text)
    return [(sentence, "") for sentence in read_m2(text)]


def read_training_pair_file(path: Path) -> list[tuple[AnnotatedSentence, str]]:
    """As :func:`read_pair_file`, for training: pairs made from an evaluation set are
    refused.
    """
    if path.suffix == PAIRS_SUFFIX and (
        pairs_origin(path.read_text(encoding="utf-8")) is Origin.EVALUATION
    ):
        msg = f"{path} was made from an evaluation set; validate or score on it instead"
        raise EvaluationDataError(msg)
    return read_pair_file(path)


@dataclass(frozen=True, slots=True)
class Pretrained:
    """Start from a pretrained encoder with fresh slot heads."""

    name: str


@dataclass(frozen=True, slots=True)
class Continued:
    """Start from a saved tagger, such as one trained on synthetic errors first."""

    path: Path


type Start = Pretrained | Continued


class Activations(StrEnum):
    """What training does with the encoder's activations between the passes."""

    KEEP = "keep"
    RECOMPUTE = "recompute"
    """Keep only each layer's input and recompute the rest in the backward pass: much
    less memory (DeBERTa-v3-large needs it at 16 GB) for about a third more time."""


class TaggerRuntime(Protocol):
    """Training and loading, which need PyTorch, kept away from the commands."""

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
        """Fine-tune the tagger ``start`` names on what ``mix`` names; save it to ``output``."""
        ...

    def load(self, path: Path, *, device: str) -> Trained:
        """The tagger saved at ``path``."""
        ...

    def train_commas(
        self,
        path: Path,
        *,
        mix: TrainingMix,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Add a comma head, fitted on what ``mix`` names, to the checkpoint at ``path``."""
        ...


class TransformerRuntime:
    """The real runtime, importing PyTorch only when a command needs it."""

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
        """Fine-tune the tagger ``start`` names on what ``mix`` names; save it to ``output``."""
        import torch
        from transformers import AutoModel

        from langcheck.synthesis.training_mix import assemble
        from langcheck.tagging.checkpoint import CheckpointInfo, load_checkpoint, save_checkpoint
        from langcheck.tagging.encoding import load_tokenizer
        from langcheck.tagging.model import SlotTagger
        from langcheck.tagging.training import InContext, TrainingOptions, train

        options = TrainingOptions(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            error_weight=error_weight,
            precision=precision,
            choice=choice,
        )
        target = torch.device(device)
        torch_limit(target)
        match start:
            case Pretrained(name=name):
                base_model = name
                tokenizer = load_tokenizer(name)
                # Some checkpoints (DeBERTa-v3) are stored in half precision; the heads and
                # training are float32.
                encoder = AutoModel.from_pretrained(name, dtype=torch.float32)
                tagger = SlotTagger(encoder).to(target)
            case Continued(path=path):
                loaded = load_checkpoint(path, device=target)
                base_model = loaded.info.base_model
                tokenizer, tagger = loaded.tokenizer, loaded.tagger
                log(f"continuing from {path} (epoch {loaded.info.epoch})")
            case _:
                assert_never(start)
        match activations:
            case Activations.KEEP:
                pass
            case Activations.RECOMPUTE:
                tagger.encoder.gradient_checkpointing_enable()
            case _:
                assert_never(activations)
        pairs, held = assemble(mix, context_sentences=context_sentences)
        training = [InContext(s, c) for s, c in pairs]
        validation = [InContext(s, c, HELD_SOURCE) for s, c in held]
        for source, found in validate_on.items():
            validation += [InContext(s, c, source) for s, c in found]
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
        return load_trained(path, device=device)

    def train_commas(
        self,
        path: Path,
        *,
        mix: TrainingMix,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Add a comma head, fitted on what ``mix`` names, to the checkpoint at ``path``."""
        import torch

        from langcheck.synthesis.training_mix import assemble
        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.comma import save_comma_head, train_comma_head
        from langcheck.tagging.training import InContext

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(path, device=target)
        pairs, _ = assemble(mix, context_sentences=loaded.info.context_sentences)
        training = [InContext(s, c) for s, c in pairs]
        log(f"fitting the comma head on {len(training)} sentences")
        head = train_comma_head(
            loaded.tagger,
            loaded.tokenizer,
            training,
            device=target,
            max_length=loaded.info.max_length,
            log=log,
        )
        save_comma_head(path, head)


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

    type CorrectedOption = Annotated[
        bool,
        Parameter(
            help="Also train on the clean version of each wrong or corrupted training "
            "sentence, with nothing flagged."
        ),
    ]
    type CleanOption = Annotated[
        Path | None,
        Parameter(help="JSONL of edited prose to corrupt with the error tables, one per text."),
    ]
    type PairsOption = Annotated[
        list[Path] | None,
        Parameter(
            help="Sentences with their edits to add, as M2 (read without context) or "
            "JSONL with context; repeatable."
        ),
    ]
    type TablesOption = Annotated[
        list[Path] | None,
        Parameter(
            help="Pairs (M2 or JSONL) to learn a second set of error tables from; each "
            "--clean sentence takes the built-in tables or these at random. Repeatable."
        ),
    ]
    type ErrorsOption = Annotated[
        list[int] | None,
        Parameter(
            help="Error counts to draw each --clean sentence's from, such as 0 1 2 3 for a "
            "quarter each; default 0 1 2 3 for the built-in tables, as the pairs show for "
            "--tables.",
            consume_multiple=True,
        ),
    ]

    def data_dir(data: Path | None) -> Path:
        return data if data is not None else env.cwd / DEFAULT_DATA

    def read_mix(
        *,
        corrected: bool,
        clean: Path | None,
        pairs: list[Path] | None,
        tables: list[Path] | None,
        errors: list[int] | None,
    ) -> TrainingMix | None:
        """The training mix the options name; None, said why, when they are not valid."""
        if any(count < 0 for count in errors or ()):
            env.err("langcheck: --errors takes counts of 0 or more")
            return None
        if clean is None and pairs is None:
            env.err("langcheck: train on --clean prose or --pairs")
            return None
        return TrainingMix(
            corrected=corrected,
            clean=tuple(read_texts(clean)) if clean is not None else (),
            pairs=tuple(pair for path in pairs or () for pair in read_training_pair_file(path)),
            tables=tuple(s for path in tables or () for s, _ in read_training_pair_file(path)),
            errors=tuple(errors or ()),
        )

    @app.command(name="train")
    def train_command(
        output: Path,
        /,
        *,
        model: Annotated[str, Parameter(help="Pretrained encoder to fine-tune.")] = DEFAULT_MODEL,
        start_from: Annotated[
            Path | None,
            Parameter(help="Continue training this saved tagger instead of --model."),
        ] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        error_weight: Annotated[
            float, Parameter(help="Weigh the loss on error slots this many times a clean one.")
        ] = 1.0,
        activations: Annotated[
            Activations, Parameter(help="recompute saves memory for larger encoders.")
        ] = Activations.KEEP,
        precision: Annotated[float, Parameter(help="Precision to choose a threshold at.")] = 0.5,
        flags_from: Annotated[
            float | None,
            Parameter(
                help="Keep the epoch whose flags from this score have the best F0.5 of "
                "precision and edits inside a flag; leave out to keep the best slot recall."
            ),
        ] = None,
        join: Annotated[
            float | None,
            Parameter(help="With --flags-from, grow flags by a falloff from this score."),
        ] = None,
        rate: Annotated[float, Parameter(help="How fast the bar for --join rises.")] = 2.0,
        context: Annotated[int, Parameter(help="Preceding sentences to read.")] = 0,
        corrected: CorrectedOption = False,
        clean: CleanOption = None,
        pairs: PairsOption = None,
        tables: TablesOption = None,
        errors: ErrorsOption = None,
        validate_on: Annotated[
            list[Path] | None,
            Parameter(
                help="Pairs (M2 or JSONL), such as realistic errors, to validate on as well as "
                "the held-out share of the training mix; epochs are judged on all of them, "
                "each set also reported. Repeatable."
            ),
        ] = None,
        device: DeviceOption = "cuda",
    ) -> int:
        """Fine-tune on --clean prose, corrupted with error tables, and on --pairs; save to
        OUTPUT. Evaluation sets are never trained on.
        """
        growth: Growth = Adjacent() if join is None else Falloff(join=join, rate=rate)
        choice: EpochChoice = (
            BySlots()
            if flags_from is None
            else ByFlags(
                TierPolicy(
                    error=max(DEFAULT_POLICY.error, flags_from),
                    suggestion=flags_from,
                    growth=growth,
                )
            )
        )

        def run() -> int:
            mix = read_mix(
                corrected=corrected,
                clean=clean,
                pairs=pairs,
                tables=tables,
                errors=errors,
            )
            if mix is None:
                return FAILED
            runtime.train(
                start=Pretrained(model) if start_from is None else Continued(start_from),
                output=output,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                error_weight=error_weight,
                activations=activations,
                precision=precision,
                choice=choice,
                context_sentences=context,
                mix=mix,
                validate_on={path.stem: read_pair_file(path) for path in validate_on or ()},
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
        corrected: CorrectedOption = False,
        clean: CleanOption = None,
        pairs: PairsOption = None,
        tables: TablesOption = None,
        errors: ErrorsOption = None,
        device: DeviceOption = "cuda",
    ) -> int:
        """Add a head to CHECKPOINT that tells missing commas from other missing words,
        fitted on the same kinds of training data as the tagger.
        """

        def run() -> int:
            mix = read_mix(
                corrected=corrected,
                clean=clean,
                pairs=pairs,
                tables=tables,
                errors=errors,
            )
            if mix is None:
                return FAILED
            runtime.train_commas(checkpoint, mix=mix, device=device, log=env.out)
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
                trained.reader,
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
