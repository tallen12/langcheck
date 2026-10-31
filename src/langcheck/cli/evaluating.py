"""``langcheck evaluate``: score the checker on annotated sets and tune the scorer.

Runs in process (like ``--local``) with the configured LLM and scorer.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from langcheck.check.prompt import PROMPT_VERSION
from langcheck.cli.checking import open_local
from langcheck.cli.environment import CliEnvironment
from langcheck.evaluation.closeness import Closeness
from langcheck.evaluation.datasets import SOURCES, Dataset, dataset_files, fetch
from langcheck.evaluation.examples import Example, load_examples
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.evaluation.report import Report, build_report, compare_reports, render_report
from langcheck.evaluation.runner import DEFAULT_CONCURRENCY, RunResult, evaluate
from langcheck.evaluation.scoring import Counts
from langcheck.evaluation.tuning import (
    DEFAULT_MAX_RECALL_LOSS,
    DEFAULT_MIN_GAIN,
    DEFAULT_MIN_SUGGESTIONS,
    Objective,
    TuningError,
    describe,
    policy_yaml,
    tune,
)
from langcheck.scoring.ranker import KEEP_ORDER
from langcheck.settings.loading import ClientSettings

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
DEFAULT_RUNS = Path("runs")

type SettingsLoader = Callable[[Path | None, Mapping[str, object]], ClientSettings]
type Guard = Callable[[Callable[[], int]], int]

type DataOption = Annotated[
    Path | None, Parameter(help="Where fetched sets are kept; default data/eval.")
]
type ConfigOption = Annotated[Path | None, Parameter(help="YAML settings file.")]


def evaluate_app(env: CliEnvironment, *, settings: SettingsLoader, guarded: Guard) -> App:
    """The ``evaluate`` commands, using the CLI's settings loading and error handling."""
    app = App(
        name="evaluate",
        help="Score the checker on annotated sets (ERRANT span-level F0.5 per kind).",
        result_action="return_value",
    )

    def data_dir(data: Path | None) -> Path:
        return data if data is not None else env.cwd / DEFAULT_DATA

    @app.command(name="fetch")
    def fetch_command(dataset: Dataset, /, *, data: DataOption = None) -> int:
        """Download DATASET once into data/eval (see its licence in the output)."""

        def run() -> int:
            files = fetch(dataset, data_dir(data), downloader=env.downloader)
            env.out(f"{dataset.value}: {SOURCES[dataset].licence}")
            for path in files:
                env.out(str(path))
            return CLEAN

        return guarded(run)

    @app.command(name="run")
    def run_command(
        dataset: Dataset,
        /,
        *,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        concurrency: Annotated[int, Parameter(help="Checks at once.")] = DEFAULT_CONCURRENCY,
        output: Annotated[Path | None, Parameter(help="Report path; default runs/.")] = None,
        data: DataOption = None,
        config: ConfigOption = None,
        llm_url: Annotated[str | None, Parameter(help="LLM API base URL.")] = None,
        llm_model: Annotated[str | None, Parameter(help="LLM model.")] = None,
        no_score: Annotated[bool, Parameter(negative=(), help="Keep the LLM's order.")] = False,
    ) -> int:
        """Check DATASET with the configured LLM and scorer, print and save the report."""

        def run() -> int:
            loaded = settings(
                config,
                {
                    "llm.base_url": llm_url,
                    "llm.model": llm_model,
                    "scorer.adapters": [] if no_score else None,
                },
            )
            files = dataset_files(dataset, data_dir(data))
            missing = [path for path in files if not path.is_file()]
            if missing:
                env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                return FAILED
            annotator = env.edit_annotators()
            examples = load_examples(dataset, files, annotator, limit=limit)
            result = asyncio.run(_run(loaded, examples, annotator, concurrency, env))
            scorer = loaded.scorer.scorer_config()
            report = build_report(
                result,
                dataset=dataset.value,
                model=loaded.llm.model,
                prompt_version=PROMPT_VERSION,
                scorer=scorer.name if scorer.adapters else KEEP_ORDER,
            )
            path = output if output is not None else _default_output(env, dataset)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            env.out(render_report(report))
            env.out(f"report: {path}")
            return CLEAN

        return guarded(run)

    @app.command
    def show(report: Path, /) -> int:
        """Print a saved report."""

        def run() -> int:
            env.out(render_report(_read(report)))
            return CLEAN

        return guarded(run)

    @app.command
    def compare(first: Path, second: Path, /) -> int:
        """Compare two reports per kind (for example two models or prompts)."""

        def run() -> int:
            env.out(compare_reports(_read(first), _read(second)))
            return CLEAN

        return guarded(run)

    @app.command(name="tune")
    def tune_command(
        report: Path,
        /,
        *,
        min_gain: Annotated[float, Parameter(help="Overall F0.5 gain needed to filter a kind.")] = (
            DEFAULT_MIN_GAIN
        ),
        max_recall_loss: Annotated[
            float, Parameter(help="Largest share of a kind's correct suggestions to drop.")
        ] = DEFAULT_MAX_RECALL_LOSS,
        min_suggestions: Annotated[
            int, Parameter(help="Scored suggestions a kind needs before it is tuned.")
        ] = DEFAULT_MIN_SUGGESTIONS,
        objective: Annotated[
            Objective, Parameter(help="Judge filters by overall F0.5 or by closeness.")
        ] = Objective.F05,
    ) -> int:
        """Choose per-kind filtering from a scored run; print the scorer.policy YAML."""

        def run() -> int:
            loaded = _read(report)
            closeness = (
                Closeness(loaded.closeness.needed, loaded.closeness.gain)
                if loaded.closeness is not None
                else None
            )
            try:
                decisions = tune(
                    loaded.outcomes,
                    Counts(loaded.overall.tp, loaded.overall.fp, loaded.overall.fn),
                    objective=objective,
                    closeness=closeness,
                    min_gain=min_gain,
                    max_recall_loss=max_recall_loss,
                    min_suggestions=min_suggestions,
                )
            except TuningError as error:
                env.err(f"langcheck: {error}")
                return FAILED
            if not decisions:
                env.err("langcheck: the report has no scored suggestions; run with a scorer")
                return FAILED
            for decision in decisions:
                env.out(describe(decision))
            env.out(policy_yaml(decisions).rstrip())
            return CLEAN

        return guarded(run)

    return app


async def _run(
    settings: ClientSettings,
    examples: Sequence[Example],
    annotator: EditAnnotator,
    concurrency: int,
    env: CliEnvironment,
) -> RunResult:
    async with open_local(settings, transport=env.transport) as local:
        return await evaluate(examples, local.checker, annotator, concurrency=concurrency)


def _read(path: Path) -> Report:
    return Report.model_validate_json(path.read_text(encoding="utf-8"))


def _default_output(env: CliEnvironment, dataset: Dataset) -> Path:
    stamp = env.clock().strftime("%Y%m%dT%H%M%S")
    return env.cwd / DEFAULT_RUNS / f"{dataset.value}-{stamp}.json"
