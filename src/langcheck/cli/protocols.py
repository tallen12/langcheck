"""Protocols of the CLI adapter."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI


class ServerRunner(Protocol):
    """Runs an ASGI app until it is stopped."""

    def run(self, app: FastAPI, *, host: str, port: int) -> None:
        """Serve ``app`` on ``host``:``port``; returns when the server stops."""
        ...


class FileWatcher(Protocol):
    """Reports when a file changes."""

    def changes(self, path: Path) -> AsyncIterator[None]:
        """Yield once after each change to ``path``, until cancelled."""
        ...
