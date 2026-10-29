"""What the CLI takes from the outside world, so commands can run against fakes."""

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from langcheck.api.startup import utc_now
from langcheck.cli.protocols import ServerRunner
from langcheck.cli.uvicorn_runner import UvicornRunner


@dataclass(frozen=True, slots=True)
class CliEnvironment:
    """The process environment, output streams and adapters the commands use.

    ``transport`` replaces the network for every HTTP client (the LLM and the service),
    for tests; ``None`` uses the real network.
    """

    environ: Mapping[str, str]
    cwd: Path
    out: Callable[[str], None]
    err: Callable[[str], None]
    runner: ServerRunner
    clock: Callable[[], datetime]
    transport: httpx.AsyncBaseTransport | None = None


def process_environment() -> CliEnvironment:
    """The real environment of this process."""
    return CliEnvironment(
        environ=dict(os.environ),
        cwd=Path.cwd(),
        out=print,
        err=lambda line: print(line, file=sys.stderr),
        runner=UvicornRunner(),
        clock=utc_now,
    )
