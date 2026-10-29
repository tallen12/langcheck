from pathlib import Path

from langcheck.cli.environment import load_edit_annotator, process_environment
from langcheck.cli.stdio_runner import StdioRunner
from langcheck.cli.uvicorn_runner import UvicornRunner
from langcheck.evaluation.datasets import HttpDownloader


def test_process_environment_uses_the_real_process() -> None:
    env = process_environment()

    assert env.cwd == Path.cwd()
    assert isinstance(env.runner, UvicornRunner)
    assert env.transport is None
    assert isinstance(env.language_servers, StdioRunner)
    assert isinstance(env.downloader, HttpDownloader)
    assert env.edit_annotators is load_edit_annotator
    assert env.clock().tzinfo is not None
