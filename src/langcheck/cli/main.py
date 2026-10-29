"""The ``langcheck`` command; with API startup, the composition root.

Commands return an exit code: 0 when the text is clean, 1 when there are suggestions or
unchecked spans, 2 when the command could not run (bad settings, unreachable server or
LLM, unreadable file).
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import yaml
from cyclopts import App, Parameter
from pydantic import ValidationError

from langcheck.api.client import ServiceError
from langcheck.api.schemas import CheckResponse, ContextBody
from langcheck.api.startup import create_app
from langcheck.auth.models import UserExistsError
from langcheck.auth.sql_store import SqlApiKeyStore
from langcheck.auth.tokens import add_user
from langcheck.cli.checking import ContextUnavailableError, open_local, open_remote
from langcheck.cli.environment import CliEnvironment, process_environment
from langcheck.cli.render import render_check, render_context
from langcheck.cli.watching import watch_file
from langcheck.database.schema import open_database
from langcheck.llm.models import LLMError
from langcheck.scoring.factory import ScorerUnavailableError
from langcheck.settings.loading import (
    ClientSettings,
    ServerSettings,
    SettingsFileError,
    load_settings,
    masked_dump,
    read_yaml,
    resolve_config_path,
)

CLEAN = 0
FOUND = 1
FAILED = 2

type ConfigOption = Annotated[
    Path | None, Parameter(help="YAML settings file; default LANGCHECK_CONFIG or ./langcheck.yaml")
]
type ServerOption = Annotated[str | None, Parameter(help="URL of a running langcheck serve.")]
type LLMUrlOption = Annotated[str | None, Parameter(help="Base URL of the OpenAI-compatible API.")]
type LLMModelOption = Annotated[str | None, Parameter(help="LLM model name.")]
type LocalOption = Annotated[
    bool, Parameter(negative=(), help="Run the checker in this process, not on the server.")
]

_EXPECTED_FAILURES = (
    SettingsFileError,
    ValidationError,
    ServiceError,
    ContextUnavailableError,
    LLMError,
    UserExistsError,
    ScorerUnavailableError,
    OSError,
)


def nested_overrides(values: Mapping[str, object]) -> dict[str, object]:
    """Settings overrides from dotted names (``llm.model``), leaving out unset (``None``)
    options so they do not mask the environment or the YAML file.
    """
    overrides: dict[str, object] = {}
    for dotted, value in values.items():
        if value is None:
            continue
        section, _, field = dotted.partition(".")
        existing = overrides.setdefault(section, {})
        if isinstance(existing, dict):
            existing[field] = value
    return overrides


def build_app(env: CliEnvironment) -> App:
    """The ``langcheck`` app with every command, running against ``env``."""
    app = App(
        name="langcheck",
        help="Check spelling and grammar with an LLM.",
        version=version("langcheck"),
        result_action="return_value",
    )
    config_app = App(name="config", help="Inspect settings.", result_action="return_value")
    admin_app = App(name="admin", help="Manage API keys.", result_action="return_value")
    app.command(config_app)
    app.command(admin_app)

    def client_settings(config: Path | None, overrides: Mapping[str, object]) -> ClientSettings:
        path = resolve_config_path(config, environ=env.environ, cwd=env.cwd)
        return load_settings(
            ClientSettings, overrides=nested_overrides(overrides), yaml_data=read_yaml(path)
        )

    def server_settings(config: Path | None, overrides: Mapping[str, object]) -> ServerSettings:
        path = resolve_config_path(config, environ=env.environ, cwd=env.cwd)
        return load_settings(
            ServerSettings, overrides=nested_overrides(overrides), yaml_data=read_yaml(path)
        )

    def guarded(run: Callable[[], int]) -> int:
        try:
            return run()
        except _EXPECTED_FAILURES as error:
            env.err(f"langcheck: {error}")
            return FAILED

    @app.command
    def check(
        file: Path,
        /,
        *,
        as_json: Annotated[bool, Parameter(name="--json", negative=())] = False,
        local: LocalOption = False,
        config: ConfigOption = None,
        server: ServerOption = None,
        llm_url: LLMUrlOption = None,
        llm_model: LLMModelOption = None,
        no_score: Annotated[
            bool, Parameter(negative=(), help="Keep the LLM's option order (--local only).")
        ] = False,
    ) -> int:
        """Check FILE and print the suggestions, or JSON with --json.

        Uses the configured server; with --local, or without a server, runs the checker
        in this process against the LLM directly, ranking options with the configured
        scorer unless --no-score is given.
        """

        def run() -> int:
            settings = client_settings(
                config,
                {
                    "server.url": server,
                    "llm.base_url": llm_url,
                    "llm.model": llm_model,
                    "scorer.adapters": [] if no_score else None,
                },
            )
            text = file.read_text(encoding="utf-8")
            uri = file.resolve().as_uri()
            response = asyncio.run(_check(settings, text, uri=uri, local=local, env=env))
            if as_json:
                env.out(response.model_dump_json(by_alias=True, indent=2))
            else:
                env.out(render_check(str(file), text, response))
            return FOUND if response.suggestions or response.unchecked else CLEAN

        return guarded(run)

    @app.command
    def watch(
        file: Path,
        /,
        *,
        local: LocalOption = False,
        config: ConfigOption = None,
        server: ServerOption = None,
        llm_url: LLMUrlOption = None,
        llm_model: LLMModelOption = None,
    ) -> int:
        """Check FILE now and after every save, printing new (+) and resolved (-)
        suggestions, until interrupted. Unchanged sentences come from the cache.
        """

        def run() -> int:
            settings = client_settings(
                config, {"server.url": server, "llm.base_url": llm_url, "llm.model": llm_model}
            )
            try:
                uri = file.resolve().as_uri()
                asyncio.run(_watch(settings, file, uri=uri, local=local, env=env))
            except KeyboardInterrupt:
                env.out("stopped")
            return CLEAN

        return guarded(run)

    @app.command(name="dict")
    def dictionary(
        file: Path,
        action: Literal["list", "add", "remove"] = "list",
        /,
        *words: str,
        config: ConfigOption = None,
        server: ServerOption = None,
    ) -> int:
        """List, add or remove words in FILE's dictionary on the server.

        Words in a document's dictionary are never flagged in it. Dictionaries are kept
        by the server, so this needs one configured (server.url or --server).
        """

        def run() -> int:
            settings = client_settings(config, {"server.url": server})
            url = settings.server.url
            if url is None:
                env.err("langcheck: dict needs a server; set server.url or pass --server")
                return FAILED
            if action != "list" and not words:
                env.err(f"langcheck: dict {action} needs at least one word")
                return FAILED
            uri = file.resolve().as_uri()
            for word in asyncio.run(_dictionary(settings, url, uri, action, words, env=env)):
                env.out(word)
            return CLEAN

        return guarded(run)

    @app.command
    def context(
        file: Path,
        /,
        *,
        as_json: Annotated[bool, Parameter(name="--json", negative=())] = False,
        local: LocalOption = False,
        config: ConfigOption = None,
        server: ServerOption = None,
        llm_url: LLMUrlOption = None,
        llm_model: LLMModelOption = None,
    ) -> int:
        """Print the document context of FILE, for tuning prompts."""

        def run() -> int:
            settings = client_settings(
                config, {"server.url": server, "llm.base_url": llm_url, "llm.model": llm_model}
            )
            text = file.read_text(encoding="utf-8")
            body = asyncio.run(_context(settings, text, local=local, env=env))
            if as_json:
                env.out(body.model_dump_json(by_alias=True, indent=2))
            else:
                env.out(render_context(body))
            return CLEAN

        return guarded(run)

    @app.command
    def serve(
        *,
        config: ConfigOption = None,
        host: Annotated[str | None, Parameter(help="Address to listen on.")] = None,
        port: Annotated[int | None, Parameter(help="Port to listen on.")] = None,
    ) -> int:
        """Run the checking service. Put a TLS reverse proxy in front of it."""

        def run() -> int:
            settings = server_settings(config, {"http.host": host, "http.port": port})
            env.runner.run(create_app(settings), host=settings.http.host, port=settings.http.port)
            return CLEAN

        return guarded(run)

    @config_app.command
    def show(
        *,
        server: Annotated[
            bool, Parameter(negative=(), help="Show the server's settings instead.")
        ] = False,
        config: ConfigOption = None,
    ) -> int:
        """Print the merged settings, with secrets masked, to see which value won."""

        def run() -> int:
            settings = server_settings(config, {}) if server else client_settings(config, {})
            env.out(yaml.safe_dump(masked_dump(settings), sort_keys=False).rstrip())
            return CLEAN

        return guarded(run)

    def key_store(config: Path | None) -> SqlApiKeyStore:
        settings = server_settings(config, {})
        return SqlApiKeyStore(open_database(settings.database.url), clock=env.clock)

    @admin_app.command(name="add-user")
    def add_user_command(name: str, /, *, config: ConfigOption = None) -> int:
        """Create an API key for NAME and print it. Only its hash is stored."""

        def run() -> int:
            env.out(add_user(key_store(config), name))
            return CLEAN

        return guarded(run)

    @admin_app.command
    def revoke(name: str, /, *, config: ConfigOption = None) -> int:
        """Remove NAME's API key."""

        def run() -> int:
            if key_store(config).revoke(name):
                env.out(f"revoked {name}")
                return CLEAN
            env.err(f"langcheck: {name} has no API key")
            return FAILED

        return guarded(run)

    @admin_app.command
    def users(*, config: ConfigOption = None) -> int:
        """List the users with an API key."""

        def run() -> int:
            for name in key_store(config).users():
                env.out(name)
            return CLEAN

        return guarded(run)

    return app


async def _check(
    settings: ClientSettings, text: str, *, uri: str, local: bool, env: CliEnvironment
) -> CheckResponse:
    url = settings.server.url
    if local or url is None:
        async with open_local(settings, transport=env.transport) as checker:
            return await checker.check(text)
    async with open_remote(settings, url, transport=env.transport) as client:
        return await client.check(text, uri=uri)


async def _watch(
    settings: ClientSettings, path: Path, *, uri: str, local: bool, env: CliEnvironment
) -> None:
    url = settings.server.url
    if local or url is None:
        async with open_local(settings, transport=env.transport) as checker:
            await watch_file(path, check=checker.check, watcher=env.watcher, out=env.out)
        return
    async with open_remote(settings, url, transport=env.transport) as client:

        async def check(text: str) -> CheckResponse:
            return await client.check(text, uri=uri)

        await watch_file(path, check=check, watcher=env.watcher, out=env.out)


async def _dictionary(
    settings: ClientSettings,
    url: str,
    uri: str,
    action: Literal["list", "add", "remove"],
    words: Sequence[str],
    *,
    env: CliEnvironment,
) -> list[str]:
    async with open_remote(settings, url, transport=env.transport) as client:
        match action:
            case "list":
                return await client.words(uri)
            case "add":
                return await client.add_words(uri, words)
            case "remove":
                return await client.remove_words(uri, words)


async def _context(
    settings: ClientSettings, text: str, *, local: bool, env: CliEnvironment
) -> ContextBody:
    url = settings.server.url
    if local or url is None:
        async with open_local(settings, transport=env.transport) as checker:
            return await checker.context(text)
    async with open_remote(settings, url, transport=env.transport) as client:
        return await client.context(text)


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``langcheck`` with ``argv``, or the process arguments, and exit with its code."""
    code = build_app(process_environment())(argv)
    raise SystemExit(code if isinstance(code, int) else CLEAN)
