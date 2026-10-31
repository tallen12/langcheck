"""Checking a text through the service or in process, with the same result type.

The CLI is a composition root: for ``--local`` it builds the same ``Checker`` the service
uses, pointed straight at the LLM.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from langcheck.api.client import ServiceClient
from langcheck.api.schemas import CheckResponse, ContextBody
from langcheck.check.checker import Checker
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.protocols import TextChecker
from langcheck.check.resolve import ValidationPolicy
from langcheck.context.builder import ContextBuilder
from langcheck.context.models import DocumentContext
from langcheck.correction.checker import TaggerLoader, open_correction_checker
from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.llm.models import ChatOptions
from langcheck.llm.structured import Failed
from langcheck.scoring.factory import AdapterLoader, load_adapter, open_ranker
from langcheck.settings.loading import ClientSettings
from langcheck.tagging.loaded import load_trained
from langcheck.text.regex_splitter import RegexSentenceSplitter

LOCAL_USER = "local"


class ContextUnavailableError(Exception):
    """The LLM gave no usable document context."""


@dataclass(frozen=True, slots=True)
class LocalChecker:
    """The checker and context builder, running in this process."""

    checker: TextChecker
    contexts: ContextBuilder

    async def check(self, text: str) -> CheckResponse:
        """Build the context (or go without it) and check ``text``."""
        built = await self.contexts.build(text)
        context = built if isinstance(built, DocumentContext) else DocumentContext.empty()
        result = await self.checker.check(text, user=LOCAL_USER, context=context)
        return CheckResponse.of(result, context)

    async def context(self, text: str) -> ContextBody:
        """The document context of ``text``."""
        match await self.contexts.build(text):
            case DocumentContext() as built:
                return ContextBody.of(built)
            case Failed(reason=reason):
                raise ContextUnavailableError(reason)


@asynccontextmanager
async def open_local(
    settings: ClientSettings,
    *,
    transport: httpx.AsyncBaseTransport | None,
    load: AdapterLoader = load_adapter,
    load_tagger: TaggerLoader = load_trained,
) -> AsyncGenerator[LocalChecker]:
    """A checker that talks to the LLM directly (and loads the configured scorer), open
    for the duration of the block.
    """
    async with (
        httpx.AsyncClient(transport=transport) as http,
        open_ranker(settings.scorer.scorer_config(), load=load) as ranker,
    ):
        llm = AsyncOpenAIChatClient(http, settings.llm.client_config())
        options = ChatOptions()
        checker: TextChecker = Checker(
            llm,
            splitter=RegexSentenceSplitter(),
            cache=MemoryResultCache(),
            model=settings.llm.model,
            options=options,
            policy=ValidationPolicy(),
            ranker=ranker,
            max_words=settings.checker.max_words,
        )
        if settings.correction.checkpoint is not None:
            checker = await open_correction_checker(
                llm,
                settings.correction,
                settings.correction.checkpoint,
                model=settings.llm.model,
                load=load_tagger,
            )
        yield LocalChecker(
            checker=checker,
            contexts=ContextBuilder(
                llm, options=options, max_paragraphs=settings.checker.context_max_paragraphs
            ),
        )


@asynccontextmanager
async def open_remote(
    settings: ClientSettings, url: str, *, transport: httpx.AsyncBaseTransport | None
) -> AsyncGenerator[ServiceClient]:
    """A client for the service at ``url``, open for the duration of the block."""
    api_key = settings.server.api_key
    async with httpx.AsyncClient(transport=transport, timeout=settings.server.timeout) as http:
        yield ServiceClient(
            http,
            base_url=url,
            api_key=api_key.get_secret_value() if api_key is not None else None,
        )
