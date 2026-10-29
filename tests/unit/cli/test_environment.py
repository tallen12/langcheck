from pathlib import Path

from langcheck.cli.environment import process_environment
from langcheck.cli.uvicorn_runner import UvicornRunner


def test_process_environment_uses_the_real_process() -> None:
    env = process_environment()

    assert env.cwd == Path.cwd()
    assert isinstance(env.runner, UvicornRunner)
    assert env.transport is None
    assert env.clock().tzinfo is not None
