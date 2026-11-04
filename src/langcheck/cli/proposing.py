"""``langcheck proposer``: train the token proposer on a tagger and measure its recall."""

import asyncio
import json
import random
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.correction.measure import CorrectionReport, Strategy
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.examples import Example, load_examples
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.proposal.measure import (
    RecallRow,
    build_cases,
    contexts_of,
    flagged_spans,
    gold_spans,
    recall_row,
    tune,
)
from langcheck.proposal.protocols import ProposalSource
from langcheck.proposal.targets import training_targets
from langcheck.synthesis.clean_text import clean_sentences, read_texts
from langcheck.synthesis.corruption import (
    corrupt_all,
    corrupt_context,
    learn_confusions,
    synthesize,
)
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import TierPolicy

if TYPE_CHECKING:
    import torch

    from langcheck.proposal.model import TokenProposer
    from langcheck.tagging.checkpoint import Checkpoint

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
TRAINING_SETS = (Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
FALSE_ALARM_POLICY = TierPolicy(suggestion=0.1)
"""Low, to find false alarms: the tagger trained on these sentences and is sure of them."""
USER_POLICY = TierPolicy()
SCORED_SET = Dataset.BEA19_DEV
SCORED_SENTENCES = 500
"""The first 500 BEA-2019 dev sentences are what earlier proposers were compared on."""

type Guard = Callable[[Callable[[], int]], int]


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


class ProposerRuntime(Protocol):
    """Training and loading, which need PyTorch, kept away from the commands."""

    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        tagger: Path,
        output: Path,
        epochs: int,
        synthetic: int,
        clean: Sequence[str],
        false_alarms: TierPolicy,
        scoring: EpochScoring | None,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Train a proposer on the ``tagger`` checkpoint and save it to ``output``, logging
        each epoch's F0.5 on ``scoring`` when given. Training sentences are flagged under
        ``false_alarms``, so the proposer learns to leave alone what it flags wrongly.
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
    """Logs how ``proposer``, as it is then, ranks right answers before any filter, and the
    scores of ``langcheck suggest --proposer``; gives the unfiltered MRR under the first
    tiers.
    """
    from langcheck.correction.measure import measure, suggest_all
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
    ]
    texts = [example.text for example in scoring.examples]
    first = scoring.tiers[0]
    contexts = contexts_of(scoring.sentences, loaded.info.context_sentences)
    # The tagger is frozen, so it reads the sentences once for every epoch.
    readings = reader.read([s.tokens for s in scoring.sentences], contexts=contexts)

    def score(epoch: int) -> float:
        cases = build_cases(
            scoring.sentences,
            flagged_spans(first),
            readings,
            contexts,
            source,
            reader,
            log=lambda _line: None,
        )
        unfiltered = recall_row("proposer", cases)
        log(unfiltered_line(epoch, first, unfiltered))
        for tiers, pipeline in pipelines:
            # One sentence at a time: the reader and the proposer share the model, unlocked.
            found = asyncio.run(suggest_all(pipeline, texts, concurrency=1, log=lambda _line: None))
            report = measure(scoring.examples, found, scoring.annotator, dataset=scoring.dataset)
            log(epoch_score_line(epoch, tiers, report))
        return unfiltered.mrr

    return score


class TransformerProposerRuntime:
    """The real runtime, importing PyTorch only when a command needs it."""

    def train(
        self,
        sentences: Sequence[AnnotatedSentence],
        *,
        tagger: Path,
        output: Path,
        epochs: int,
        synthetic: int,
        clean: Sequence[str],
        false_alarms: TierPolicy,
        scoring: EpochScoring | None,
        device: str,
        log: Callable[[str], None],
    ) -> None:
        """Train a proposer on the ``tagger`` checkpoint and save it to ``output``, logging
        each epoch's F0.5 on ``scoring`` when given. Training sentences are flagged under
        ``false_alarms``, so the proposer learns to leave alone what it flags wrongly.
        """
        import torch

        from langcheck.proposal.checkpoint import (
            ProposerInfo,
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
        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.loaded import torch_limit
        from langcheck.tagging.tiers import slot_flags
        from langcheck.tagging.training import split_holdout

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(tagger, device=target)
        reader = _tagger_reader(tagger, loaded, target)
        contexts = contexts_of(sentences, loaded.info.context_sentences)

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

        real = list(zip(sentences, contexts, strict=True))
        kept, held = split_holdout(real)
        training = items_of([s for s, _ in kept], [c for _, c in kept])
        validation = items_of([s for s, _ in held], [c for _, c in held])
        if synthetic:
            made = synthesize([s for s, _ in kept], copies=synthetic)
            training += items_of([s for _, s in made], [kept[i][1] for i, _ in made])
        if clean:
            found = list(clean_sentences(clean, context_sentences=loaded.info.context_sentences))
            confusions = learn_confusions([s for s, _ in kept])
            rng = random.Random(0)
            training += items_of(
                corrupt_all([c.words for c in found], confusions),
                [corrupt_context(c.context, confusions, rng) for c in found],
            )
        cases = sum(len(item.targets) for item in training)
        log(f"{len(training)} training sentences, {cases} targets")
        shape = shape_for(loaded.tagger)
        proposer = TokenProposer(shape).to(target)
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
                options=ProposerOptions(epochs=epochs),
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
        synthetic: Annotated[
            int, Parameter(help="Corrupted copies of each corrected training sentence to add.")
        ] = 0,
        clean: Annotated[
            Path | None, Parameter(help="JSONL of edited prose to corrupt and add, one per text.")
        ] = None,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        scored: Annotated[
            int,
            Parameter(help="BEA-2019 dev sentences to measure F0.5 on after each epoch; 0 skips."),
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
        device: str = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Train on the BEA-2019 W&I+LOCNESS and FCE training sets; save to OUTPUT."""

        def run() -> int:
            sentences: list[AnnotatedSentence] = []
            for dataset in TRAINING_SETS:
                found = read_set(dataset, data)
                if found is None:
                    return FAILED
                sentences += found
            scoring: EpochScoring | None = None
            if scored > 0:
                files = dataset_files(SCORED_SET, data_dir(data))
                if not all(path.is_file() for path in files):
                    env.err(f"langcheck: run `langcheck evaluate fetch {SCORED_SET.value}` first")
                    return FAILED
                annotator = env.edit_annotators()
                gold = read_set(SCORED_SET, data)
                if gold is None:
                    return FAILED
                scoring = EpochScoring(
                    SCORED_SET.value,
                    tuple(load_examples(SCORED_SET, files, annotator, limit=scored)),
                    annotator,
                    tuple(gold[:scored]),
                    tuple(
                        TierPolicy(error=max(USER_POLICY.error, at), suggestion=at)
                        for at in suggestion
                    ),
                )
            runtime.train(
                sentences[:limit],
                tagger=tagger,
                output=output,
                epochs=epochs,
                synthetic=synthetic,
                clean=read_texts(clean) if clean is not None else (),
                false_alarms=TierPolicy(
                    error=max(USER_POLICY.error, false_alarms), suggestion=false_alarms
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
        tune_on: Annotated[Dataset, Parameter(help="Half tunes, half tests.")] = Dataset.BEA19_DEV,
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
            for dataset in (tune_on, test_on):
                found = read_set(dataset, data)
                if found is None:
                    return FAILED
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
                    if dataset is tune_on and name == "flags":
                        half = len(sentences) // 2
                        weight, penalty = tune([c for c in cases if c.sentence < half])
                        env.out(f"tuned: weight {weight} penalty {penalty}")
                    if dataset is tune_on:
                        half = len(sentences) // 2
                        held = [c for c in cases if c.sentence >= half]
                    alarms = [c for c in cases if c.false_alarm]
                    rows = [
                        recall_row(f"{dataset.value} {name} proposer", held),
                        recall_row(
                            f"{dataset.value} {name} re-ranked",
                            held,
                            weight=weight,
                            penalty=penalty,
                        ),
                        recall_row(f"{dataset.value} {name} tagger only", held, weight=1e3),
                    ]
                    env.out(render_rows(rows))
                    env.out(
                        f"{dataset.value} {name}: {len(alarms)} false alarms, "
                        f"{sum(c.keep_first for c in alarms)} keep first; "
                        f"{sum(c.keep_first for c in cases if not c.false_alarm)} of "
                        f"{sum(not c.false_alarm for c in cases)} real errors keep first"
                    )
                    saved[f"{dataset.value} {name}"] = [row.model_dump() for row in rows]
            if output is not None:
                output.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            return CLEAN

        return guarded(run)

    return app
