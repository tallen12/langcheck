"""Protocols of the CLI adapter."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from pygls.lsp.server import LanguageServer


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


class LanguageServerRunner(Protocol):
    """Runs a language server over the process's stdin and stdout."""

    def run(self, server: LanguageServer) -> None:
        """Serve until the editor shuts the server down."""
        ...
