"""``langcheck synth``: write errors into clean prose, drawn as a labeled set shows them.

Errors drawn from one distribution and scored on another show whether a model learned
errors or only its training noise. Tables learned from an evaluation set mark what they
make as evaluation data, which training refuses.
"""

import dataclasses
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from langcheck.cli.environment import CliEnvironment
from langcheck.cli.tagging import read_pair_file
from langcheck.evaluation.context_pairs import Origin, pairs_origin, write_pairs
from langcheck.evaluation.datasets import SOURCES, Dataset, Format, dataset_files
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
from langcheck.synthesis.clean_text import clean_sentences, read_texts
from langcheck.synthesis.corruption import Confusions, corrupt_in_context, table_sets
from langcheck.synthesis.default_tables import default_confusions

CLEAN = 0
FAILED = 2
DEFAULT_DATA = Path("data") / "eval"
PAIRS_SUFFIX = ".jsonl"

type Guard = Callable[[Callable[[], int]], int]


def synth_app(env: CliEnvironment, *, guarded: Guard) -> App:
    """The ``synth`` commands."""
    app = App(
        name="synth",
        help="Write synthetic errors into clean prose from error tables.",
        result_action="return_value",
    )

    @app.command(name="corrupt")
    def corrupt_command(
        *,
        clean: Annotated[Path, Parameter(help="JSONL of edited prose, one text per line.")],
        output: Annotated[Path, Parameter(help="JSONL of the corrupted sentences.")],
        tables: Annotated[
            list[Path] | None,
            Parameter(
                help="Labeled pairs (M2 or JSONL) to learn errors from, a set of tables "
                "each; repeatable. With neither this nor --from-set, the built-in tables."
            ),
        ] = None,
        from_set: Annotated[
            list[Dataset] | None,
            Parameter(
                help="Fetched M2 evaluation sets to learn errors from; what they make is "
                "marked as evaluation data. Repeatable."
            ),
        ] = None,
        errors: Annotated[
            list[int] | None,
            Parameter(
                help="Error counts to draw from, such as 0 1 2 3 for a quarter each; "
                "default as each source shows.",
                consume_multiple=True,
            ),
        ] = None,
        copies: Annotated[int, Parameter(help="Corrupted versions of each sentence.")] = 1,
        context: Annotated[int, Parameter(help="Preceding sentences to keep.")] = 2,
        limit: Annotated[int | None, Parameter(help="Only the first N sentences.")] = None,
        seed: int = 0,
        data: Annotated[Path | None, Parameter(help="Default data/eval.")] = None,
    ) -> int:
        """Write each sentence of CLEAN with errors drawn from the tables, keeping its
        context; the edits that undo them are its gold.
        """

        def run() -> int:
            if any(count < 0 for count in errors or ()) or copies < 1:
                env.err("langcheck: --errors takes counts of 0 or more, --copies 1 or more")
                return FAILED
            sources: list[list[AnnotatedSentence]] = []
            origin = Origin.OPEN
            for path in tables or ():
                sources.append([s for s, _ in read_pair_file(path)])
                text = path.read_text(encoding="utf-8")
                if path.suffix == PAIRS_SUFFIX and pairs_origin(text) is Origin.EVALUATION:
                    origin = Origin.EVALUATION
            for dataset in from_set or ():
                if SOURCES[dataset].format is not Format.M2:
                    env.err(f"langcheck: {dataset.value} has no typed edits to learn from")
                    return FAILED
                path = dataset_files(dataset, data if data is not None else env.cwd / DEFAULT_DATA)[
                    0
                ]
                if not path.is_file():
                    env.err(f"langcheck: run `langcheck evaluate fetch {dataset.value}` first")
                    return FAILED
                sources.append(read_m2(path.read_text(encoding="utf-8")))
                origin = Origin.EVALUATION
            chosen: tuple[Confusions, ...] = (
                table_sets(sources) if sources else (default_confusions(),)
            )
            if errors is not None:
                chosen = tuple(dataclasses.replace(t, error_counts=tuple(errors)) for t in chosen)
            prose = clean_sentences(read_texts(clean), context_sentences=context)
            found = [(c.words, c.context) for c in prose][:limit]
            made = corrupt_in_context(found * copies, chosen, seed=seed)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(write_pairs(made, origin=origin), encoding="utf-8")
            codes = Counter(edit.code for sentence, _ in made for edit in sentence.annotators[0])
            top = ", ".join(f"{code} {n}" for code, n in codes.most_common(5))
            env.out(f"{len(made)} sentences, {codes.total()} errors ({top}); {origin} data")
            env.out(f"saved: {output}")
            return CLEAN

        return guarded(run)

    return app
