"""Protocols of the CLI adapter."""

from typing import Protocol

from fastapi import FastAPI


class ServerRunner(Protocol):
    """Runs an ASGI app until it is stopped."""

    def run(self, app: FastAPI, *, host: str, port: int) -> None:
        """Serve ``app`` on ``host``:``port``; returns when the server stops."""
        ...
