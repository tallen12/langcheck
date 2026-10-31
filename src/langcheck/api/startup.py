"""The server's composition root: builds the services from ``ServerSettings``."""

import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import assert_never

import httpx
from fastapi import FastAPI
from sqlalchemy import Engine

from langcheck.api.app import Services, build_app
from langcheck.auth.authenticators import ApiKeyAuthenticator, TrustingAuthenticator
from langcheck.auth.protocols import Authenticator
from langcheck.auth.sql_store import SqlApiKeyStore
from langcheck.check.checker import Checker
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.protocols import TextChecker
from langcheck.check.resolve import ValidationPolicy
from langcheck.context.builder import ContextBuilder, NoContext
from langcheck.correction.checker import (
    ProposerLoader,
    TaggerLoader,
    open_correction_checker,
    open_proposing_checker,
)
from langcheck.database.schema import open_database
from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.dictionary.protocols import DictionaryStore
from langcheck.dictionary.routing import UntitledInMemory
from langcheck.dictionary.sql_store import SqlDictionaryStore
from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.sessions import DocumentSessions
from langcheck.limits.user_limiter import UserLimiter
from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.llm.models import ChatOptions
from langcheck.proposal.loaded import load_proposing
from langcheck.scoring.factory import AdapterLoader, load_adapter, open_ranker
from langcheck.settings.loading import ServerSettings
from langcheck.settings.models import AuthMode, DictionaryBackend
from langcheck.tagging.loaded import load_trained
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
    load_tagger: TaggerLoader = load_trained,
    load_proposer: ProposerLoader = load_proposing,
) -> AsyncGenerator[Services]:
    """Open the LLM client, scorer, database and document sessions for ``settings`` and
    close them afterwards.

    ``transport`` replaces the network for the LLM client, and ``load`` and ``load_tagger``
    the model loaders, for tests.
    """
    engine = open_database(settings.database.url)
    try:
        async with (
            httpx.AsyncClient(transport=transport) as http,
            open_ranker(settings.scorer.scorer_config(), load=load) as ranker,
        ):
            llm = AsyncOpenAIChatClient(http, settings.llm.client_config())
            options = ChatOptions()
            splitter = RegexSentenceSplitter()
            checker: TextChecker = Checker(
                llm,
                splitter=splitter,
                cache=MemoryResultCache(),
                model=settings.llm.model,
                options=options,
                policy=ValidationPolicy(),
                ranker=ranker,
                max_words=settings.checker.max_words,
            )
            checkpoint = settings.correction.checkpoint
            if checkpoint is not None and settings.correction.proposer is not None:
                checker = await open_proposing_checker(
                    settings.correction,
                    checkpoint,
                    settings.correction.proposer,
                    load=load_proposer,
                )
            elif checkpoint is not None:
                checker = await open_correction_checker(
                    llm,
                    settings.correction,
                    checkpoint,
                    model=settings.llm.model,
                    load=load_tagger,
                )
            contexts = ContextBuilder(
                llm, options=options, max_paragraphs=settings.checker.context_max_paragraphs
            )
            # The correction checker reads no context, so checks need not ask the LLM for one.
            check_contexts = NoContext() if settings.correction.checkpoint is not None else contexts
            limiter = UserLimiter(
                max_in_flight=settings.limits.in_flight,
                per_minute=settings.limits.per_minute,
                clock=time.monotonic,
            )
            dictionary = _dictionary(settings, engine)
            documents = DocumentSessions(
                MemoryDocumentStore(),
                checker=checker,
                contexts=check_contexts,
                splitter=splitter,
                limiter=limiter,
                dictionary=dictionary,
                debounce=settings.checker.debounce,
                refresh_share=settings.checker.context_refresh_share,
            )
            try:
                yield Services(
                    authenticator=_authenticator(settings, SqlApiKeyStore(engine, clock=clock)),
                    checker=checker,
                    contexts=contexts,
                    limiter=limiter,
                    documents=documents,
                    dictionary=dictionary,
                    check_contexts=check_contexts,
                )
            finally:
                await documents.aclose()
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


def _dictionary(settings: ServerSettings, engine: Engine) -> DictionaryStore:
    match settings.dictionary.backend:
        case DictionaryBackend.SQL:
            persistent: DictionaryStore = SqlDictionaryStore(engine)
        case DictionaryBackend.MEMORY:
            persistent = MemoryDictionaryStore()
        case unreachable:
            assert_never(unreachable)
    return UntitledInMemory(persistent, scratch=MemoryDictionaryStore())
