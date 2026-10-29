"""What the CLI takes from the outside world, so commands can run against fakes."""

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from langcheck.api.startup import utc_now
from langcheck.cli.file_watcher import WatchfilesWatcher
from langcheck.cli.protocols import FileWatcher, LanguageServerRunner, ServerRunner
from langcheck.cli.stdio_runner import StdioRunner
from langcheck.cli.uvicorn_runner import UvicornRunner
from langcheck.evaluation.datasets import Downloader, HttpDownloader
from langcheck.evaluation.protocols import EditAnnotator


@dataclass(frozen=True, slots=True)
class CliEnvironment:
    """The process environment, output streams and adapters the commands use.

    ``transport`` replaces the network for every HTTP client (the LLM and the service),
    for tests; ``None`` uses the real network. ``edit_annotators`` loads ERRANT only when
    an evaluation needs it.
    """

    environ: Mapping[str, str]
    cwd: Path
    out: Callable[[str], None]
    err: Callable[[str], None]
    runner: ServerRunner
    watcher: FileWatcher
    language_servers: LanguageServerRunner
    downloader: Downloader
    edit_annotators: Callable[[], EditAnnotator]
    clock: Callable[[], datetime]
    transport: httpx.AsyncBaseTransport | None = None


DOWNLOAD_TIMEOUT = 120.0


def load_edit_annotator() -> EditAnnotator:
    """ERRANT on spaCy, imported here so other commands never load them."""
    from langcheck.evaluation.errant_annotator import load_errant_annotator

    return load_errant_annotator()


def process_environment() -> CliEnvironment:
    """The real environment of this process."""
    return CliEnvironment(
        environ=dict(os.environ),
        cwd=Path.cwd(),
        out=print,
        err=lambda line: print(line, file=sys.stderr),
        runner=UvicornRunner(),
        watcher=WatchfilesWatcher(),
        language_servers=StdioRunner(),
        downloader=HttpDownloader(httpx.Client(timeout=DOWNLOAD_TIMEOUT)),
        edit_annotators=load_edit_annotator,
        clock=utc_now,
    )
