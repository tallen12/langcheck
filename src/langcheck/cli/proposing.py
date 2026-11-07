"""``langcheck proposer``: train the token proposer on a tagger and measure its recall."""

import asyncio
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, assert_never

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.cli.tagging import read_pair_file
from langcheck.correction.measure import CorrectionReport, Strategy
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.examples import Example, annotated_examples
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.proposal.measure import (
    RecallRow,
    build_cases,
    contexts_of,
    flagged_spans,
    gold_spans,
    raw_cases,
    recall_row,
    tune,
)
from langcheck.proposal.protocols import ProposalSource
from langcheck.proposal.targets import training_targets
from langcheck.synthesis.clean_text import read_texts
from langcheck.synthesis.training_mix import TrainingMix
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import Adjacent, Falloff, Growth, TierPolicy

if TYPE_CHECKING:
    import torch

    from langcheck.proposal.model import TokenProposer
    from langcheck.tagging.checkpoint import Checkpoint

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
FALSE_ALARM_POLICY = TierPolicy(suggestion=0.1)
"""Low, to find false alarms: the tagger trained on these sentences and is sure of them."""
USER_POLICY = TierPolicy()
SCORED_SENTENCES = 500
"""Sentences of --score-with measured after each epoch."""
SCORED_BATCH = 64
"""Texts corrected together when scoring an epoch."""
DEFAULT_LEARNING_RATE = 3e-4
"""As ``ProposerOptions`` has it, kept here so the command needs no PyTorch."""

type Guard = Callable[[Callable[[], int]], int]


class Measured(StrEnum):
    """What is scored after each epoch."""

    PROPOSALS = "proposals"
    """Only the beam's own proposals at the flags, before any reranking or filter."""
    PIPELINE = "pipeline"
    """Those, then the whole correction pipeline's corrections at each threshold."""


@dataclass(frozen=True, slots=True)
class EpochScoring:
    """Sentences scored after each epoch: the proposer's own proposals at the flags, then
    the whole correction pipeline's corrections.
    """

    dataset: str
    examples: tuple[Example, ...]
    annotator: EditAnnotator
    sentences: tuple[AnnotatedSentence, ...]
    """The same sentences with their gold edits, for the proposals before any filter."""
    tiers: tuple[TierPolicy, ...] = (USER_POLICY,)
    """Each is scored on its own, to compare where flagging begins. The first chooses the
    epoch kept: the one whose proposals, before reranking or filtering, put a right answer
    highest (MRR). Training targets the right answer first; the filters come afterwards."""
    measured: Measured = Measured.PIPELINE
    extra: tuple[tuple[str, tuple[tuple[AnnotatedSentence, str], ...]], ...] = ()
    """More sets, by name, with each sentence's context: their unfiltered proposals are
    scored too, and with any the epoch is chosen by the MRR of all sets' spans together."""


def unfiltered_line(epoch: int, tiers: TierPolicy, row: RecallRow) -> str:
    """One epoch's proposals at the flags, in the proposer's own order, before filtering."""
    cells = " ".join(f"@{k} {v:.3f}" for k, v in row.recall.items())
    return (
        f"epoch {epoch}: unfiltered proposals at flags from {tiers.suggestion:.2f}, "
        f"{row.cases} spans with an error, MRR {row.mrr:.3f}, recall {cells}"
    )


def epoch_score_line(epoch: int, tiers: TierPolicy, report: CorrectionReport) -> str:
    """One epoch's scores in the order ``langcheck suggest`` offers corrections, then how
    far down the offered list the right answers sit.
    """
    rows = {row.strategy: row for row in report.strategies}
    row = rows[Strategy.TAGGER_RANKED]
    return (
        f"epoch {epoch}: {report.dataset} {report.sentences} sentences, flags from "
        f"{tiers.suggestion:.2f}, tagger-ranked "
        f"precision {row.precision:.3f} recall {row.recall:.3f} F0.5 {row.f05:.4f} "
        f"gain {row.gain}/{row.needed}, MRR {report.mrr:.3f}, recall in top 3 "
        f"{rows[Strategy.BEST_OF_3].recall:.3f} top 5 {rows[Strategy.BEST_OF_5].recall:.3f} "
        f"any {rows[Strategy.BEST_OFFERED].recall:.3f}"
    )


@dataclass(frozen=True, slots=True)
class Fresh:
    """Start a new proposer."""


@dataclass(frozen=True, slots=True)
class FromProposer:
    """Start from a saved proposer, such as one trained on synthetic errors first."""

    path: Path


type ProposerStart = Fresh | FromProposer


class ProposerRuntime(Protocol):
    """Training and loading, which need PyTorch, kept away from the commands."""

    def train(
        self,
        *,
        tagger: Path,
        output: Path,
        start: ProposerStart,
        epochs: int,
        learning_rate: float,
        mix: TrainingMix,
        false_alarms: TierPolicy,
        scoring: EpochScoring | None,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Train a proposer on the ``tagger`` checkpoint and what ``mix`` names, and save
        it to ``output``, logging each epoch's F0.5 on ``scoring`` when given. Training
        sentences are flagged under ``false_alarms``, so the proposer learns to leave alone
        what it flags wrongly.
        """
        ...

    def load(
        self, tagger: Path, proposer: Path, *, device: str, width: int
    ) -> tuple[ProposalSource, SentenceReader, int]:
        """A proposal source, the tagger's reader and how many sentences it reads before."""
        ...


def _tagger_reader(path: Path, loaded: Checkpoint, device: torch.device) -> SentenceReader:
    """A reader sharing ``loaded``'s model, since a proposer reads with the same encoder."""
    from langcheck.tagging.comma import has_comma_head, load_comma_head
    from langcheck.tagging.inference import TaggerReader

    head = load_comma_head(path, loaded.tagger, device=device) if has_comma_head(path) else None
    return TaggerReader(
        loaded.tagger,
        loaded.tokenizer,
        device=device,
        max_length=loaded.info.max_length,
        comma_head=head,
    )


def _epoch_scorer(
    scoring: EpochScoring,
    loaded: Checkpoint,
    reader: SentenceReader,
    proposer: TokenProposer,
    device: torch.device,
    log: Callable[[str], None],
) -> Callable[[int], float]:
    """Logs how ``proposer``, as it is then, ranks right answers before any filter, and,
    when the pipeline is measured, the scores of ``langcheck suggest --proposer``; gives the
    unfiltered MRR under the first tiers.
    """
    from langcheck.correction.measure import measure, suggest_batched
    from langcheck.correction.models import CorrectionPolicy
    from langcheck.correction.pipeline import CorrectionPipeline
    from langcheck.proposal.corrector import ProposerCorrector, proposer_policy
    from langcheck.proposal.loaded import BEAM_WIDTH
    from langcheck.proposal.search import BeamProposer

    policy = proposer_policy(CorrectionPolicy())
    source = BeamProposer(
        loaded.tagger,
        proposer,
        loaded.tokenizer,
        device=device,
        max_length=loaded.info.max_length,
        width=BEAM_WIDTH,
    )
    pipelines = [
        (
            tiers,
            CorrectionPipeline(
                corrector=ProposerCorrector(source, reader, policy),
                reader=reader,
                tiers=tiers,
                policy=policy,
                context_sentences=loaded.info.context_sentences,
            ),
        )
        for tiers in scoring.tiers
        if scoring.measured is Measured.PIPELINE
    ]
    texts = [example.text for example in scoring.examples]
    first = scoring.tiers[0]
    contexts = contexts_of(scoring.sentences, loaded.info.context_sentences)
    # The tagger is frozen, so it reads the sentences once for every epoch.
    readings = reader.read([s.tokens for s in scoring.sentences], contexts=contexts)

    extra = [
        (
            name,
            [sentence for sentence, _ in found],
            [context for _, context in found],
            reader.read(
                [sentence.tokens for sentence, _ in found],
                contexts=[context for _, context in found],
            ),
        )
        for name, found in scoring.extra
    ]

    def score(epoch: int) -> float:
        cases = raw_cases(scoring.sentences, flagged_spans(first), readings, contexts, source)
        unfiltered = recall_row("proposer", cases)
        log(unfiltered_line(epoch, first, unfiltered))
        together = list(cases)
        for name, sentences, more_contexts, more_readings in extra:
            more = raw_cases(sentences, flagged_spans(first), more_readings, more_contexts, source)
            log(f"{unfiltered_line(epoch, first, recall_row('proposer', more))} on {name}")
            together += more
        for tiers, pipeline in pipelines:
            # One batch at a time: the reader and the proposer share the model, unlocked.
            found = asyncio.run(
                suggest_batched(pipeline, texts, batch=SCORED_BATCH, log=lambda _line: None)
            )
            report = measure(scoring.examples, found, scoring.annotator, dataset=scoring.dataset)
            log(epoch_score_line(epoch, tiers, report))
        if not extra:
            return unfiltered.mrr
        combined = recall_row("proposer", together)
        log(f"{unfiltered_line(epoch, first, combined)} on all sets")
        return combined.mrr

    return score


class TransformerProposerRuntime:
    """The real runtime, importing PyTorch only when a command needs it."""

    def train(
        self,
        *,
        tagger: Path,
        output: Path,
        start: ProposerStart,
        epochs: int,
        learning_rate: float,
        mix: TrainingMix,
        false_alarms: TierPolicy,
        scoring: EpochScoring | None,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Train a proposer on the ``tagger`` checkpoint and what ``mix`` names, and save
        it to ``output``, logging each epoch's F0.5 on ``scoring`` when given. Training
        sentences are flagged under ``false_alarms``, so the proposer learns to leave alone
        what it flags wrongly.
        """
        import torch

        from langcheck.proposal.checkpoint import (
            ProposerInfo,
            load_proposer,
            save_proposer,
            shape_for,
        )
        from langcheck.proposal.model import TokenProposer
        from langcheck.proposal.stores import MappedFileStore
        from langcheck.proposal.training import (
            HighestScore,
            LowestLoss,
            ProposerItem,
            ProposerOptions,
            train_proposer,
        )
        from langcheck.synthesis.training_mix import assemble
        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.loaded import torch_limit
        from langcheck.tagging.tiers import slot_flags

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(tagger, device=target)
        reader = _tagger_reader(tagger, loaded, target)

        def items_of(found: Sequence[AnnotatedSentence], read: Sequence[str]) -> list[ProposerItem]:
            log(f"flagging {len(found)} sentences")
            readings = reader.read([s.tokens for s in found], contexts=read)
            return [
                ProposerItem(
                    sentence.tokens,
                    context,
                    training_targets(
                        sentence, slot_flags(reading.scores, reading.commas, false_alarms)
                    ),
                )
                for sentence, context, reading in zip(found, read, readings, strict=True)
            ]

        kept, held = assemble(mix, context_sentences=loaded.info.context_sentences)

        def flagged(found: Sequence[tuple[AnnotatedSentence, str]]) -> list[ProposerItem]:
            return items_of([s for s, _ in found], [c for _, c in found])

        training = flagged(kept)
        validation = flagged(held)
        cases = sum(len(item.targets) for item in training)
        log(f"{len(training)} training sentences, {cases} targets")
        shape = shape_for(loaded.tagger)
        match start:
            case Fresh():
                proposer = TokenProposer(shape).to(target)
            case FromProposer(path=path):
                proposer, info = load_proposer(path, device=target)
                if (info.encoder_size, info.embedding_size, info.vocab_size) != (
                    shape.encoder_size,
                    shape.embedding_size,
                    shape.vocab_size,
                ):
                    msg = f"the proposer at {path} reads another kind of tagger than {tagger}"
                    raise ValueError(msg)
                log(f"continuing from {path} (epoch {info.epoch})")
            case _:
                assert_never(start)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Beside the output rather than in the temp folder, which may be RAM-backed.
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}-states-", dir=output.parent
        ) as states:
            result = train_proposer(
                loaded.tagger,
                proposer,
                loaded.tokenizer,
                training,
                validation,
                device=target,
                options=ProposerOptions(epochs=epochs, learning_rate=learning_rate),
                log=log,
                store=MappedFileStore(Path(states)),
                choice=(
                    LowestLoss()
                    if scoring is None
                    else HighestScore(_epoch_scorer(scoring, loaded, reader, proposer, target, log))
                ),
            )
        save_proposer(
            output,
            proposer,
            ProposerInfo.describing(
                shape,
                tagger=str(tagger),
                train_sentences=len(training),
                validation_sentences=len(validation),
                epoch=result.epoch,
                validation_loss=result.loss,
            ),
        )

    def load(
        self, tagger: Path, proposer: Path, *, device: str, width: int
    ) -> tuple[ProposalSource, SentenceReader, int]:
        """A proposal source, the tagger's reader and how many sentences it reads before."""
        import torch

        from langcheck.proposal.checkpoint import load_proposer
        from langcheck.proposal.search import BeamProposer
        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.loaded import torch_limit

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(tagger, device=target)
        model, _ = load_proposer(proposer, device=target)
        source = BeamProposer(
            loaded.tagger,
            model,
            loaded.tokenizer,
            device=target,
            max_length=loaded.info.max_length,
            width=width,
        )
        return source, _tagger_reader(tagger, loaded, target), loaded.info.context_sentences


def render_rows(rows: Sequence[RecallRow]) -> str:
    """``rows`` as a table."""
    lines = []
    for row in rows:
        cells = " ".join(f"@{k} {v:.3f}" for k, v in row.recall.items())
        lines.append(
            f"{row.scoring:40} n={row.cases:5} {cells} MRR {row.mrr:.3f} "
            f"deletions-first {row.deletions_first:.3f}"
        )
    return "\n".join(lines)


def proposer_app(env: CliEnvironment, *, runtime: ProposerRuntime, guarded: Guard) -> App:
    """The ``proposer`` commands, around ``runtime``."""
    app = App(
        name="proposer",
        help="Train the token proposer and measure its recall.",
        result_action="return_value",
    )

    def data_dir(data: Path | None) -> Path:
        return data if data is not None else env.cwd / DEFAULT_DATA

    def read_set(dataset: Dataset, data: Path | None) -> list[AnnotatedSentence] | None:
        [path] = dataset_files(dataset, data_dir(data))
        if not path.is_file():
            env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
            return None
        return read_m2(path.read_text(encoding="utf-8"))

    @app.command(name="train")
    def train_command(
        output: Path,
        /,
        *,
        tagger: Annotated[Path, Parameter(help="The tagger checkpoint to read with.")],
        epochs: int = 3,
        start_from: Annotated[
            Path | None,
            Parameter(help="Continue a saved proposer, such as one trained on synthetic errors."),
        ] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        corrected: Annotated[
            bool,
            Parameter(
                help="Also train on the clean version of each wrong or corrupted training "
                "sentence, where the tagger's false alarms should be left alone."
            ),
        ] = False,
        clean: Annotated[
            Path | None,
            Parameter(help="JSONL of edited prose to corrupt with the error tables, one per text."),
        ] = None,
        pairs: Annotated[
            list[Path] | None,
            Parameter(
                help="Sentences with their edits to add, as M2 (read without context) or "
                "JSONL with context; repeatable."
            ),
        ] = None,
        tables: Annotated[
            list[Path] | None,
            Parameter(
                help="Pairs (M2 or JSONL) to learn a second set of error tables from; each "
                "--clean sentence takes the built-in tables or these at random. Repeatable."
            ),
        ] = None,
        errors: Annotated[
            list[int] | None,
            Parameter(
                help="Error counts to draw each --clean sentence's from, such as 0 1 2 3 for "
                "a quarter each; default 0 1 2 3 for the built-in tables, as the pairs show "
                "for --tables.",
                consume_multiple=True,
            ),
        ] = None,
        score_with: Annotated[
            Path | None,
            Parameter(
                help="Pairs (M2 or JSONL), not an evaluation set, to measure F0.5 on after "
                "each epoch and choose the epoch by; leave out to keep the lowest loss."
            ),
        ] = None,
        scored: Annotated[
            int, Parameter(help="Sentences of --score-with to measure.")
        ] = SCORED_SENTENCES,
        suggestion: Annotated[
            tuple[float, ...],
            Parameter(
                help="Score where flagging begins when measuring F0.5; repeat to measure several."
            ),
        ] = (USER_POLICY.suggestion,),
        false_alarms: Annotated[
            float,
            Parameter(help="Score where training sentences are flagged, false alarms included."),
        ] = FALSE_ALARM_POLICY.suggestion,
        join: Annotated[
            float | None,
            Parameter(
                help="Lowest score of a word next to a flag that the flag takes in, rising "
                "by --rate for each word further; leave out to join only flagged slots."
            ),
        ] = None,
        rate: Annotated[float, Parameter(help="How fast the bar for --join rises.")] = 2.0,
        measured: Annotated[
            Measured, Parameter(help="Score only the raw proposals, or the pipeline as well.")
        ] = Measured.PIPELINE,
        score_on: Annotated[
            list[Path] | None,
            Parameter(
                help="Pairs (M2 or JSONL), such as a held-out set, whose raw proposals are all "
                "scored after each epoch too; the epoch is then chosen on all sets. Repeatable."
            ),
        ] = None,
        device: str = "cuda",
    ) -> int:
        """Train on --clean prose, corrupted with error tables, and on --pairs; save to
        OUTPUT. Evaluation sets are never trained on. Flags for training and scoring both
        grow as --join and --rate say.
        """
        growth: Growth = Adjacent() if join is None else Falloff(join=join, rate=rate)

        def run() -> int:
            if clean is None and pairs is None:
                env.err("langcheck: train on --clean prose or --pairs")
                return FAILED
            if any(count < 0 for count in errors or ()):
                env.err("langcheck: --errors takes counts of 0 or more")
                return FAILED
            scoring: EpochScoring | None = None
            if score_with is not None:
                annotator = env.edit_annotators()
                gold = [sentence for sentence, _ in read_pair_file(score_with)][:scored]
                scoring = EpochScoring(
                    score_with.stem,
                    tuple(annotated_examples(gold, annotator)),
                    annotator,
                    tuple(gold),
                    tuple(
                        TierPolicy(error=max(USER_POLICY.error, at), suggestion=at, growth=growth)
                        for at in suggestion
                    ),
                    measured,
                    tuple((path.stem, tuple(read_pair_file(path))) for path in score_on or ()),
                )
            elif score_on is not None:
                env.err("langcheck: --score-on needs --score-with")
                return FAILED
            runtime.train(
                tagger=tagger,
                output=output,
                start=Fresh() if start_from is None else FromProposer(start_from),
                epochs=epochs,
                learning_rate=learning_rate,
                mix=TrainingMix(
                    corrected=corrected,
                    clean=tuple(read_texts(clean)) if clean is not None else (),
                    pairs=tuple(pair for path in pairs or () for pair in read_pair_file(path)),
                    tables=tuple(s for path in tables or () for s, _ in read_pair_file(path)),
                    errors=tuple(errors or ()),
                ),
                false_alarms=TierPolicy(
                    error=max(USER_POLICY.error, false_alarms),
                    suggestion=false_alarms,
                    growth=growth,
                ),
                scoring=scoring,
                device=device,
                log=env.out,
            )
            env.out(f"saved: {output}")
            return CLEAN

        return guarded(run)

    @app.command(name="measure")
    def measure_command(
        proposer: Path,
        /,
        *,
        tagger: Annotated[Path, Parameter(help="The tagger checkpoint it was trained on.")],
        tune_on: Annotated[
            Path,
            Parameter(help="Pairs (M2 or JSONL), not an evaluation set: half tunes, half tests."),
        ],
        test_on: Annotated[Dataset, Parameter(help="Tested with those weights.")] = (
            Dataset.CONLL14_TEST
        ),
        width: Annotated[int, Parameter(help="Beam width.")] = 16,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        output: Annotated[Path | None, Parameter(help="Also save the rows here.")] = None,
        device: str = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Recall@k at real flags and gold spans, then re-ranked by the tagger."""

        def run() -> int:
            source, reader, context = runtime.load(tagger, proposer, device=device, width=width)
            saved: dict[str, list[dict[str, object]]] = {}
            weight, penalty = 0.0, 0.0
            tested = read_set(test_on, data)
            if tested is None:
                return FAILED
            sets = (
                (tune_on.stem, [sentence for sentence, _ in read_pair_file(tune_on)], True),
                (test_on.value, tested, False),
            )
            for set_name, found, tunes in sets:
                sentences = found[:limit]
                contexts = contexts_of(sentences, context)
                readings = reader.read([s.tokens for s in sentences], contexts=contexts)
                for name, spans in (
                    ("flags", flagged_spans(USER_POLICY)),
                    ("gold spans", gold_spans(sentences)),
                ):
                    cases = build_cases(
                        sentences, spans, readings, contexts, source, reader, log=env.out
                    )
                    held = cases
                    if tunes and name == "flags":
                        half = len(sentences) // 2
                        weight, penalty = tune([c for c in cases if c.sentence < half])
                        env.out(f"tuned: weight {weight} penalty {penalty}")
                    if tunes:
                        half = len(sentences) // 2
                        held = [c for c in cases if c.sentence >= half]
                    alarms = [c for c in cases if c.false_alarm]
                    rows = [
                        recall_row(f"{set_name} {name} proposer", held),
                        recall_row(
                            f"{set_name} {name} re-ranked",
                            held,
                            weight=weight,
                            penalty=penalty,
                        ),
                        recall_row(f"{set_name} {name} tagger only", held, weight=1e3),
                    ]
                    env.out(render_rows(rows))
                    env.out(
                        f"{set_name} {name}: {len(alarms)} false alarms, "
                        f"{sum(c.keep_first for c in alarms)} keep first; "
                        f"{sum(c.keep_first for c in cases if not c.false_alarm)} of "
                        f"{sum(not c.false_alarm for c in cases)} real errors keep first"
                    )
                    saved[f"{set_name} {name}"] = [row.model_dump() for row in rows]
            if output is not None:
                output.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            return CLEAN

        return guarded(run)

    return app
