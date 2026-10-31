"""``langcheck suggest``: flag a text with the tagger and have the LLM correct the flags."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.cli.tagging import DEFAULT_DATA, DEFAULT_POLICY, TaggerRuntime, Trained, render_flag
from langcheck.correction.measure import measure, render_measurement, suggest_all
from langcheck.correction.models import (
    Corrected,
    CorrectionPolicy,
    FlagOutcome,
    Ranked,
    Rejected,
    Unavailable,
)
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.evaluation.datasets import Dataset, dataset_files
from langcheck.evaluation.examples import Example, load_corrected_pairs, load_examples
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.llm.models import ChatOptions
from langcheck.settings.loading import ClientSettings
from langcheck.tagging.tiers import TierPolicy

CLEAN = 0
FOUND = 1
FAILED = 2

DEFAULT_CORRECTIONS = CorrectionPolicy()

type SettingsLoader = Callable[[Path | None, Mapping[str, object]], ClientSettings]
type Guard = Callable[[Callable[[], int]], int]


def render_ranked(place: int, ranked: Ranked) -> str:
    """One offered correction: its reduction, confidence, edit, kind and explanation."""
    candidate = ranked.candidate
    edit = f"{candidate.original!r} -> {candidate.replacement.strip()!r}"
    return (
        f"  {place}.\t-{ranked.reduction:.3f}\t{ranked.confidence.value}\t{edit}"
        f"\t{candidate.kind.value}"
        f"\t{candidate.explanation}"
    )


def render_rejected(rejected: Rejected) -> str:
    """One correction not offered and why."""
    reduction = "" if rejected.reduction is None else f"{-rejected.reduction:+.3f}"
    edit = f"{rejected.original!r} -> {rejected.replacement!r}"
    return f"  x\t{reduction}\t{edit}\t{rejected.reason.value}"


def render_outcome(text: str, outcome: FlagOutcome, *, show_rejected: bool) -> list[str]:
    """A flag's line, followed by a line per correction."""
    match outcome:
        case Unavailable(flag=flag, reason=reason):
            return [render_flag(text, flag), f"  unavailable: {reason}"]
        case Corrected(flag=flag, ranked=ranked, rejected=rejected):
            lines = [render_flag(text, flag)]
            lines += [render_ranked(i, r) for i, r in enumerate(ranked, start=1)]
            if show_rejected:
                lines += [render_rejected(r) for r in rejected]
            return lines


def suggest_app(
    env: CliEnvironment, *, runtime: TaggerRuntime, settings: SettingsLoader, guarded: Guard
) -> App:
    """The ``suggest`` command."""
    app = App(
        name="suggest",
        help="Flag errors with the tagger, correct them with the LLM and rank the corrections.",
        result_action="return_value",
    )

    @asynccontextmanager
    async def open_pipeline(
        trained: Trained, loaded: ClientSettings, tiers: TierPolicy, policy: CorrectionPolicy
    ) -> AsyncGenerator[CorrectionPipeline]:
        async with httpx.AsyncClient(transport=env.transport) as http:
            yield CorrectionPipeline(
                llm=AsyncOpenAIChatClient(http, loaded.llm.client_config()),
                scorer=trained.scorer,
                commas=trained.commas,
                tiers=tiers,
                policy=policy,
                options=ChatOptions(),
                context_sentences=trained.context_sentences,
            )

    async def outcomes(
        text: str,
        trained: Trained,
        loaded: ClientSettings,
        tiers: TierPolicy,
        policy: CorrectionPolicy,
    ) -> list[FlagOutcome]:
        async with open_pipeline(trained, loaded, tiers, policy) as pipeline:
            return await pipeline.suggest(text)

    async def outcomes_of_all(
        texts: Sequence[str],
        trained: Trained,
        loaded: ClientSettings,
        tiers: TierPolicy,
        policy: CorrectionPolicy,
    ) -> list[list[FlagOutcome]]:
        async with open_pipeline(trained, loaded, tiers, policy) as pipeline:
            return await suggest_all(
                pipeline, texts, concurrency=loaded.llm.concurrency, log=env.err
            )

    def report_on(
        name: str,
        examples: Sequence[Example],
        annotator: EditAnnotator,
        trained: Trained,
        loaded: ClientSettings,
        policy: CorrectionPolicy,
        output: Path | None,
    ) -> None:
        texts = [example.text for example in examples]
        found = asyncio.run(outcomes_of_all(texts, trained, loaded, DEFAULT_POLICY, policy))
        report = measure(examples, found, annotator, dataset=name)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        env.out(render_measurement(report))

    @app.default
    def suggest(
        checkpoint: Path,
        text: str,
        /,
        *,
        count: Annotated[int, Parameter(help="Corrections to ask for per flag.")] = (
            DEFAULT_CORRECTIONS.count
        ),
        min_reduction: Annotated[
            float, Parameter(help="How much a correction must lower its sentence's score.")
        ] = DEFAULT_CORRECTIONS.min_reduction,
        confident_reduction: Annotated[
            float, Parameter(help="How much a correction must lower it for high confidence.")
        ] = DEFAULT_CORRECTIONS.confident_reduction,
        error: Annotated[float, Parameter(help="Score where the error tier begins.")] = (
            DEFAULT_POLICY.error
        ),
        suggestion: Annotated[float, Parameter(help="Lowest score flagged at all.")] = (
            DEFAULT_POLICY.suggestion
        ),
        show_rejected: Annotated[
            bool, Parameter(negative=(), help="Also print corrections not offered, and why.")
        ] = False,
        device: Annotated[str, Parameter(help="PyTorch device, such as cpu or cuda.")] = "cuda",
        config: Annotated[Path | None, Parameter(help="YAML settings file.")] = None,
        llm_url: Annotated[
            str | None, Parameter(help="Base URL of the OpenAI-compatible API.")
        ] = None,
        llm_model: Annotated[str | None, Parameter(help="LLM model name.")] = None,
    ) -> int:
        """Print each flag of TEXT with its corrections, the one lowering the score most first."""

        def run() -> int:
            loaded = settings(config, {"llm.base_url": llm_url, "llm.model": llm_model})
            trained = runtime.load(checkpoint, device=device)
            found = asyncio.run(
                outcomes(
                    text,
                    trained,
                    loaded,
                    TierPolicy(error=error, suggestion=suggestion),
                    CorrectionPolicy(
                        count=count,
                        min_reduction=min_reduction,
                        confident_reduction=confident_reduction,
                    ),
                )
            )
            for outcome in found:
                for line in render_outcome(text, outcome, show_rejected=show_rejected):
                    env.out(line)
            return FOUND if found else CLEAN

        return guarded(run)

    @app.command(name="measure")
    def measure_command(
        checkpoint: Path,
        dataset: Dataset,
        /,
        *,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = 200,
        output: Annotated[Path | None, Parameter(help="Also save the report here.")] = None,
        count: Annotated[int, Parameter(help="Corrections to ask for per flag.")] = (
            DEFAULT_CORRECTIONS.count
        ),
        min_reduction: Annotated[
            float, Parameter(help="How much a correction must lower its sentence's score.")
        ] = DEFAULT_CORRECTIONS.min_reduction,
        device: Annotated[str, Parameter(help="PyTorch device, such as cpu or cuda.")] = "cuda",
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
        config: Annotated[Path | None, Parameter(help="YAML settings file.")] = None,
        llm_url: Annotated[
            str | None, Parameter(help="Base URL of the OpenAI-compatible API.")
        ] = None,
        llm_model: Annotated[str | None, Parameter(help="LLM model name.")] = None,
    ) -> int:
        """Score corrections and each filter against DATASET's references."""

        def run() -> int:
            files = dataset_files(dataset, data if data is not None else env.cwd / DEFAULT_DATA)
            if not all(path.is_file() for path in files):
                env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                return FAILED
            loaded = settings(config, {"llm.base_url": llm_url, "llm.model": llm_model})
            trained = runtime.load(checkpoint, device=device)
            annotator = env.edit_annotators()
            examples = load_examples(dataset, files, annotator, limit=limit)
            policy = CorrectionPolicy(count=count, min_reduction=min_reduction)
            report_on(dataset.value, examples, annotator, trained, loaded, policy, output)
            return CLEAN

        return guarded(run)

    @app.command(name="measure-file")
    def measure_file_command(
        checkpoint: Path,
        file: Path,
        /,
        *,
        limit: Annotated[int | None, Parameter(help="Only the first N lines.")] = 200,
        output: Annotated[Path | None, Parameter(help="Also save the report here.")] = None,
        count: Annotated[int, Parameter(help="Corrections to ask for per flag.")] = (
            DEFAULT_CORRECTIONS.count
        ),
        min_reduction: Annotated[
            float, Parameter(help="How much a correction must lower its sentence's score.")
        ] = DEFAULT_CORRECTIONS.min_reduction,
        device: Annotated[str, Parameter(help="PyTorch device, such as cpu or cuda.")] = "cuda",
        config: Annotated[Path | None, Parameter(help="YAML settings file.")] = None,
        llm_url: Annotated[
            str | None, Parameter(help="Base URL of the OpenAI-compatible API.")
        ] = None,
        llm_model: Annotated[str | None, Parameter(help="LLM model name.")] = None,
    ) -> int:
        """Like ``measure``, on a JSONL FILE with ``text`` and ``corrected`` on each line."""

        def run() -> int:
            loaded = settings(config, {"llm.base_url": llm_url, "llm.model": llm_model})
            trained = runtime.load(checkpoint, device=device)
            annotator = env.edit_annotators()
            examples = load_corrected_pairs(file, annotator, limit=limit)
            policy = CorrectionPolicy(count=count, min_reduction=min_reduction)
            report_on(file.stem, examples, annotator, trained, loaded, policy, output)
            return CLEAN

        return guarded(run)

    return app
