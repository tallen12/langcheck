"""``ServerRunner`` with uvicorn."""

import uvicorn
from fastapi import FastAPI


class UvicornRunner:
    """Serves the app in one uvicorn process, as the plan's single-process design needs."""

    def run(self, app: FastAPI, *, host: str, port: int) -> None:
        """Serve ``app`` until interrupted."""
        uvicorn.run(app, host=host, port=port)
