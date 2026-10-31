"""``langcheck proposer``: train the token proposer on a tagger and measure its recall."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
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
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import TierPolicy

if TYPE_CHECKING:
    import torch

    from langcheck.tagging.checkpoint import Checkpoint

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
TRAINING_SETS = (Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN)
FALSE_ALARM_POLICY = TierPolicy(suggestion=0.1)
"""Low, to find false alarms: the tagger trained on these sentences and is sure of them."""
USER_POLICY = TierPolicy()

type Guard = Callable[[Callable[[], int]], int]


class ProposerRuntime(Protocol):
    """Training and loading, which need PyTorch, kept away from the commands."""

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
        """Train a proposer on the ``tagger`` checkpoint and save it to ``output``."""
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


class TransformerProposerRuntime:
    """The real runtime, importing PyTorch only when a command needs it."""

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
        """Train a proposer on the ``tagger`` checkpoint and save it to ``output``."""
        import torch

        from langcheck.proposal.checkpoint import (
            ProposerInfo,
            save_proposer,
            shape_for,
        )
        from langcheck.proposal.model import TokenProposer
        from langcheck.proposal.training import ProposerItem, ProposerOptions, train_proposer
        from langcheck.tagging.checkpoint import load_checkpoint
        from langcheck.tagging.loaded import torch_limit
        from langcheck.tagging.tiers import slot_flags
        from langcheck.tagging.training import split_holdout

        target = torch.device(device)
        torch_limit(target)
        loaded = load_checkpoint(tagger, device=target)
        reader = _tagger_reader(tagger, loaded, target)
        contexts = contexts_of(sentences, loaded.info.context_sentences)
        log(f"flagging {len(sentences)} sentences")
        readings = reader.read([s.tokens for s in sentences], contexts=contexts)
        items = [
            ProposerItem(
                sentence.tokens,
                context,
                training_targets(
                    sentence, slot_flags(reading.scores, reading.commas, FALSE_ALARM_POLICY)
                ),
            )
            for sentence, context, reading in zip(sentences, contexts, readings, strict=True)
        ]
        training, validation = split_holdout(items)
        cases = sum(len(item.targets) for item in training)
        log(f"{len(training)} training sentences, {cases} targets")
        shape = shape_for(loaded.tagger)
        proposer = TokenProposer(shape).to(target)
        result = train_proposer(
            loaded.tagger,
            proposer,
            loaded.tokenizer,
            training,
            validation,
            device=target,
            options=ProposerOptions(epochs=epochs),
            log=log,
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
            f"{row.scoring:40} n={row.cases:5} {cells} deletions-first {row.deletions_first:.3f}"
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
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
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
            runtime.train(
                sentences[:limit],
                tagger=tagger,
                output=output,
                epochs=epochs,
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
