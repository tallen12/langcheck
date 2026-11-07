"""``langcheck judge``: have a panel of LLMs judge a set's gold edits and a run's edits.

The panel runs on the configured LLM server, one model after another. Verdicts go to a
JSONL store per set under ``data/judgments``, which is never committed, so a later run
only judges edits the store lacks.
"""

import asyncio
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.examples import CorrectedPair, Example, load_examples
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.judging.consensus import DEFAULT_QUORUM
from langcheck.judging.llm_judge import LLMEditJudge, judge_name
from langcheck.judging.models import Item, Source
from langcheck.judging.panel import (
    DEFAULT_CONCURRENCY,
    JudgeRun,
    Summary,
    Tally,
    pool_item,
    run_panel,
)
from langcheck.judging.panel import summarize as summarize_panel
from langcheck.judging.roster import DEFAULT_PANEL, DEFAULT_STANDBY, choose_panel
from langcheck.judging.store import JsonlJudgmentStore, store_path
from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.settings.loading import ClientSettings

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
DEFAULT_STORE = Path("data") / "judgments"

type SettingsLoader = Callable[[Path | None, Mapping[str, object]], ClientSettings]
type Guard = Callable[[Callable[[], int]], int]

type ModelsOption = Annotated[
    list[str] | None,
    Parameter(help=f"Judge models; default {' '.join(DEFAULT_PANEL)}.", consume_multiple=True),
]
type CorrectorOption = Annotated[
    str | None,
    Parameter(
        help=f"Model whose edits are judged; a judge of its family gives way to "
        f"--standby ({DEFAULT_STANDBY})."
    ),
]
type StandbyOption = Annotated[str, Parameter(help="Judge that replaces the corrector's family.")]
type QuorumOption = Annotated[int, Parameter(help="Judges that must agree on a verdict.")]
type CorrectionsOption = Annotated[
    list[Path] | None,
    Parameter(help="JSONL of text and corrected, from a run on the set; repeatable."),
]
type ThinkOption = Annotated[
    bool, Parameter(negative=(), help="Judges reason before answering: slower, stored apart.")
]
type LimitOption = Annotated[int | None, Parameter(help="Only the first N sentences.")]
type StoreOption = Annotated[Path | None, Parameter(help="Default data/judgments.")]
type DataOption = Annotated[Path | None, Parameter(help="Default data/eval.")]


def judge_app(env: CliEnvironment, *, settings: SettingsLoader, guarded: Guard) -> App:
    """The ``judge`` commands."""
    app = App(
        name="judge",
        help="Judge edits with a panel of LLMs instead of trusting gold alone.",
        result_action="return_value",
    )

    def items_for(
        dataset: Dataset,
        corrections: Sequence[Path],
        limit: int | None,
        data: Path | None,
    ) -> list[Item] | None:
        files = dataset_files(dataset, data if data is not None else env.cwd / DEFAULT_DATA)
        if any(not path.is_file() for path in files):
            env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
            return None
        annotator = env.edit_annotators()
        examples = load_examples(dataset, files, annotator, limit=limit)
        return pooled(examples, corrections, annotator, env.out)

    def store_for(dataset: Dataset, store: Path | None) -> JsonlJudgmentStore:
        root = store if store is not None else env.cwd / DEFAULT_STORE
        return JsonlJudgmentStore(store_path(root, dataset.value))

    @app.command(name="run")
    def run_command(
        dataset: Dataset,
        /,
        *,
        models: ModelsOption = None,
        corrector: CorrectorOption = None,
        standby: StandbyOption = DEFAULT_STANDBY,
        quorum: QuorumOption = DEFAULT_QUORUM,
        corrections: CorrectionsOption = None,
        limit: LimitOption = None,
        concurrency: Annotated[int, Parameter(help="Requests at once.")] = DEFAULT_CONCURRENCY,
        think: ThinkOption = False,
        store: StoreOption = None,
        data: DataOption = None,
        config: Annotated[Path | None, Parameter(help="YAML settings file.")] = None,
        llm_url: Annotated[str | None, Parameter(help="LLM API base URL.")] = None,
    ) -> int:
        """Judge DATASET's gold edits and any run's edits; print the panel's consensus."""

        def run() -> int:
            panel = choose_panel(models or DEFAULT_PANEL, corrector=corrector, standby=standby)
            if not 1 <= quorum <= len(panel) or concurrency < 1:
                env.err(f"langcheck: --quorum takes 1 to {len(panel)}, --concurrency 1 or more")
                return FAILED
            items = items_for(dataset, corrections or (), limit, data)
            if items is None:
                return FAILED
            env.out(f"panel: {', '.join(panel)} (quorum {quorum})")
            loaded = settings(config, {"llm.base_url": llm_url})
            judgments = store_for(dataset, store)
            runs = asyncio.run(
                _judge(
                    loaded,
                    panel,
                    items,
                    judgments,
                    concurrency=concurrency,
                    think=think,
                    env=env,
                )
            )
            for result in runs:
                line = f"{result.judge}: judged {result.judged}, already had {result.skipped}"
                if result.failures:
                    line += f", failed {len(result.failures)} ({result.failures[0]})"
                env.out(line)
            names = [judge_name(model, think=think) for model in panel]
            summary = summarize_panel(items, judgments.load(), names, quorum=quorum)
            env.out(render_summary(summary))
            env.out(f"judgments: {store_path(store or env.cwd / DEFAULT_STORE, dataset.value)}")
            return FAILED if any(result.failures for result in runs) else CLEAN

        return guarded(run)

    @app.command
    def show(
        dataset: Dataset,
        /,
        *,
        models: ModelsOption = None,
        corrector: CorrectorOption = None,
        standby: StandbyOption = DEFAULT_STANDBY,
        quorum: QuorumOption = DEFAULT_QUORUM,
        corrections: CorrectionsOption = None,
        limit: LimitOption = None,
        think: ThinkOption = False,
        store: StoreOption = None,
        data: DataOption = None,
    ) -> int:
        """Print the panel's consensus on DATASET from stored judgments, judging nothing."""

        def run() -> int:
            panel = choose_panel(models or DEFAULT_PANEL, corrector=corrector, standby=standby)
            if not 1 <= quorum <= len(panel):
                env.err(f"langcheck: --quorum takes 1 to {len(panel)}")
                return FAILED
            items = items_for(dataset, corrections or (), limit, data)
            if items is None:
                return FAILED
            names = [judge_name(model, think=think) for model in panel]
            summary = summarize_panel(items, store_for(dataset, store).load(), names, quorum=quorum)
            env.out(render_summary(summary))
            return CLEAN

        return guarded(run)

    return app


async def _judge(
    settings: ClientSettings,
    panel: Sequence[str],
    items: Sequence[Item],
    store: JsonlJudgmentStore,
    *,
    concurrency: int,
    think: bool,
    env: CliEnvironment,
) -> list[JudgeRun]:
    config = dataclasses.replace(settings.llm.client_config(), concurrency=concurrency)
    async with httpx.AsyncClient(transport=env.transport) as http:
        judges = [
            LLMEditJudge(
                AsyncOpenAIChatClient(http, dataclasses.replace(config, model=model)),
                model,
                think=think,
            )
            for model in panel
        ]
        return await run_panel(items, judges, store, concurrency=concurrency, progress=env.out)


def pooled(
    examples: Sequence[Example],
    corrections: Sequence[Path],
    annotator: EditAnnotator,
    out: Callable[[str], None],
) -> list[Item]:
    """Each example with its gold corrections and each corrections file's, named after
    the file.

    A corrections line belongs to the example whose checked text it holds; lines matching
    no example are counted and left out.
    """
    runs: list[tuple[str, dict[str, str]]] = []
    texts = {example.text for example in examples}
    for path in corrections:
        found: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pair = CorrectedPair.model_validate_json(line)
                found[pair.text] = pair.corrected
        unmatched = sum(text not in texts for text in found)
        if unmatched:
            out(f"{path}: {unmatched} of {len(found)} lines match no sentence of the set")
        runs.append((path.stem, found))
    return [
        pool_item(
            example.source,
            gold=example.annotators,
            runs=[
                (name, annotator.system_edits(example.source, run[example.text]))
                for name, run in runs
                if example.text in run
            ],
            written=example.written,
        )
        for example in examples
    ]


def render_summary(summary: Summary) -> str:
    """The panel's consensus as a few lines of text."""
    lines: list[str] = []
    for source in Source:
        if summary.edits[source].total:
            verdicts = ("valid", "neutral", "wrong")
            lines.append(_tally_line(f"{source} edits", summary.edits[source], verdicts))
            lines.append(f"  linked, non-local: {summary.linked[source]}")
    if summary.preferences.total:
        ranked = tuple(label for label, _ in summary.preferences.agreed.most_common())
        lines.append(_tally_line("best correction", summary.preferences, ranked))
    lines.append(_tally_line("error left in sentence", summary.remaining, ("True", "False")))
    for judge, counts in summary.by_judge.items():
        shown = ", ".join(f"{verdict} {n}" for verdict, n in sorted(counts.items()))
        lines.append(f"  {judge}: {shown or 'nothing judged'}")
    for (first, second), share in summary.agreement.items():
        lines.append(f"  agreement {first} / {second}: {share:.0%}")
    return "\n".join(lines)


def _tally_line(label: str, tally: Tally, order: Sequence[str]) -> str:
    names = {"True": "yes", "False": "no"}
    parts = [f"{names.get(value, value)} {tally.agreed[value]}" for value in order]
    parts += [f"disputed {tally.disputed}", f"pending {tally.pending}"]
    return f"{label}: {tally.total} ({', '.join(parts)})"
