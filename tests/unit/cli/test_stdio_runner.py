import pytest
from pygls.lsp.server import LanguageServer

from langcheck.cli.protocols import LanguageServerRunner
from langcheck.cli.stdio_runner import StdioRunner


def test_runs_the_server_on_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[LanguageServer] = []
    server = LanguageServer("x", "1")
    monkeypatch.setattr(server, "start_io", lambda: started.append(server))
    runner: LanguageServerRunner = StdioRunner()

    runner.run(server)

    assert started == [server]
