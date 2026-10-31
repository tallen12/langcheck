"""The FastAPI app, built from already constructed services."""

import asyncio
import math
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from langcheck.api.documents import document_router
from langcheck.api.schemas import (
    CheckRequest,
    CheckResponse,
    ContextBody,
    ContextRequest,
    HealthResponse,
)
from langcheck.auth.models import Unauthenticated, User
from langcheck.auth.protocols import Authenticator
from langcheck.check.protocols import TextChecker
from langcheck.context.models import DocumentContext
from langcheck.context.protocols import ContextSource
from langcheck.dictionary.protocols import DictionaryStore
from langcheck.documents.protocols import DocumentService
from langcheck.limits.models import Permit, RateLimited
from langcheck.limits.protocols import CheckLimiter
from langcheck.llm.structured import Failed


@dataclass(frozen=True, slots=True)
class Services:
    """Everything the endpoints need, built by the composition root."""

    authenticator: Authenticator
    checker: TextChecker
    contexts: ContextSource
    limiter: CheckLimiter
    documents: DocumentService
    dictionary: DictionaryStore
    check_contexts: ContextSource | None = None
    """The context ``/check`` builds when the client sends none, if not from ``contexts``;
    ``/context`` always uses ``contexts``.
    """


type ServicesFactory = Callable[[], AbstractAsyncContextManager[Services]]


def build_app(services: ServicesFactory) -> FastAPI:
    """The app. ``services`` opens the services when the app starts and closes them when
    it stops, so clients and connections live exactly as long as the server.
    """
    holder: list[Services] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with services() as opened:
            holder.append(opened)
            try:
                yield
            finally:
                holder.clear()

    def current() -> Services:
        return holder[0]

    async def user(request: Request) -> User:
        headers = {name.lower(): value for name, value in request.headers.items()}
        match await current().authenticator.authenticate(headers):
            case User() as found:
                return found
            case Unauthenticated(reason=reason):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=reason,
                    headers={"WWW-Authenticate": "Bearer"},
                )

    app = FastAPI(title="langcheck", lifespan=lifespan)
    app.include_router(
        document_router(lambda: current().documents, lambda: current().dictionary, user)
    )

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/check")
    async def check(body: CheckRequest, caller: Annotated[User, Depends(user)]) -> CheckResponse:
        services = current()
        match services.limiter.admit(caller.name):
            case RateLimited(retry_after=wait):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="too many checks; try again later",
                    headers={"Retry-After": str(math.ceil(wait))},
                )
            case Permit() as permit:
                with permit:
                    return await _check(services, body, caller)

    @app.post("/context")
    async def context(body: ContextRequest, caller: Annotated[User, Depends(user)]) -> ContextBody:
        match await current().contexts.build(body.text, dictionary=body.dictionary):
            case DocumentContext() as built:
                return ContextBody.of(built)
            case Failed(reason=reason):
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=reason)

    return app


async def _check(services: Services, body: CheckRequest, caller: User) -> CheckResponse:
    words = list(body.dictionary)
    if body.uri is not None:
        words += await asyncio.to_thread(services.dictionary.words, caller.name, body.uri)
    if body.context is not None:
        context = body.context.to_context()
    else:
        # A one-shot check builds its context every time, so the stored dictionary stays
        # out of it: in the glossary it would change the context version, and with it
        # every cache key, whenever a word is added or removed.
        contexts = services.check_contexts or services.contexts
        context = await _context_or_empty(contexts, body.text, body.dictionary)
    result = await services.checker.check(
        body.text, user=caller.name, context=context, dictionary=words
    )
    return CheckResponse.of(result, context)


async def _context_or_empty(
    contexts: ContextSource, text: str, dictionary: list[str]
) -> DocumentContext:
    """A check goes ahead without context rather than failing when it cannot be built."""
    match await contexts.build(text, dictionary=dictionary):
        case DocumentContext() as built:
            return built
        case Failed():
            return DocumentContext.empty()
