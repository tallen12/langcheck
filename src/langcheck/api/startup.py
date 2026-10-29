"""The server's composition root: builds the services from ``ServerSettings``."""

import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import assert_never

import httpx
from fastapi import FastAPI

from langcheck.api.app import Services, build_app
from langcheck.auth.authenticators import ApiKeyAuthenticator, TrustingAuthenticator
from langcheck.auth.protocols import Authenticator
from langcheck.auth.sql_store import SqlApiKeyStore
from langcheck.check.checker import Checker
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.resolve import ValidationPolicy
from langcheck.context.builder import ContextBuilder
from langcheck.database.schema import open_database
from langcheck.limits.user_limiter import UserLimiter
from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.llm.models import ChatOptions
from langcheck.scoring.factory import AdapterLoader, load_adapter, open_ranker
from langcheck.settings.loading import ServerSettings
from langcheck.settings.models import AuthMode
from langcheck.text.regex_splitter import RegexSentenceSplitter


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


@asynccontextmanager
async def open_services(
    settings: ServerSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], datetime] = utc_now,
    load: AdapterLoader = load_adapter,
) -> AsyncGenerator[Services]:
    """Open the LLM client, scorer and database for ``settings`` and close them afterwards.

    ``transport`` replaces the network for the LLM client and ``load`` the scorer model
    loader, for tests.
    """
    engine = open_database(settings.database.url)
    try:
        async with (
            httpx.AsyncClient(transport=transport) as http,
            open_ranker(settings.scorer.scorer_config(), load=load) as ranker,
        ):
            llm = AsyncOpenAIChatClient(http, settings.llm.client_config())
            options = ChatOptions()
            yield Services(
                authenticator=_authenticator(settings, SqlApiKeyStore(engine, clock=clock)),
                checker=Checker(
                    llm,
                    splitter=RegexSentenceSplitter(),
                    cache=MemoryResultCache(),
                    model=settings.llm.model,
                    options=options,
                    policy=ValidationPolicy(),
                    ranker=ranker,
                    max_words=settings.checker.max_words,
                ),
                contexts=ContextBuilder(
                    llm,
                    options=options,
                    max_paragraphs=settings.checker.context_max_paragraphs,
                ),
                limiter=UserLimiter(
                    max_in_flight=settings.limits.in_flight,
                    per_minute=settings.limits.per_minute,
                    clock=time.monotonic,
                ),
            )
    finally:
        engine.dispose()


def create_app(
    settings: ServerSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """The app for ``settings``; its services open when the server starts."""
    return build_app(lambda: open_services(settings, transport=transport))


def _authenticator(settings: ServerSettings, store: SqlApiKeyStore) -> Authenticator:
    match settings.auth.mode:
        case AuthMode.API_KEY:
            return ApiKeyAuthenticator(store)
        case AuthMode.NONE:
            return TrustingAuthenticator(proxy_user_header=settings.auth.proxy_user_header)
        case unreachable:
            assert_never(unreachable)
