from fastapi import FastAPI

from langcheck.cli.protocols import ServerRunner


class NullRunner:
    """Runs nothing."""

    def __init__(self) -> None:
        self.ports: list[int] = []

    def run(self, app: FastAPI, *, host: str, port: int) -> None:
        self.ports.append(port)


def test_server_runner_contract() -> None:
    runner = NullRunner()
    contract: ServerRunner = runner

    contract.run(FastAPI(), host="127.0.0.1", port=8000)

    assert runner.ports == [8000]
