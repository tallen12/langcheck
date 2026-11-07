"""Build the trace page: the template in ``assets/`` with a trace from ``trace.py`` inside.

Run from the repository root:

    uv run python .agents/skills/model-trace/scripts/build_page.py trace.json --out page.html
"""

import argparse
import html
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "model-trace.html"
TITLE = "__TITLE__"
DATA = "/*DATA*/"


class _Summary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    dataset: str
    proposer: str


class _Trace(BaseModel):
    """Only what naming the page needs; the page reads the rest."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    summary: _Summary


class _Arguments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace: Path
    out: Path
    title: str | None


def page(trace: str, *, title: str | None = None) -> str:
    """The template filled with ``trace``, a JSON document from ``trace.py``."""
    summary = _Trace.model_validate_json(trace).summary
    name = title or f"{summary.dataset} {summary.proposer} trace"
    # Inside a script element only "</" can end it early; JSON allows "<\/" for it.
    data = json.dumps(json.loads(trace), separators=(",", ":")).replace("</", "<\\/")
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.replace(TITLE, html.escape(name)).replace(DATA, data)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", help="Default '<set> <proposer> trace'.")
    arguments = _Arguments.model_validate(vars(parser.parse_args(argv)))
    built = page(arguments.trace.read_text(encoding="utf-8"), title=arguments.title)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(built, encoding="utf-8")
    print(f"page: {arguments.out} ({len(built.encode()) // 1024} KB)")


if __name__ == "__main__":
    main()
