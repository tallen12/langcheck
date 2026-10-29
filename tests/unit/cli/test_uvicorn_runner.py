import pytest
import uvicorn
from fastapi import FastAPI

from langcheck.cli.uvicorn_runner import UvicornRunner


def test_runs_the_app_with_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, str, int]] = []

    def fake_run(app: object, *, host: str, port: int) -> None:
        calls.append((app, host, port))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    app = FastAPI()

    UvicornRunner().run(app, host="0.0.0.0", port=8123)

    assert calls == [(app, "0.0.0.0", 8123)]
