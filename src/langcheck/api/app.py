"""The FastAPI app, built from already constructed services."""

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

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
from langcheck.llm.structured import Failed


@dataclass(frozen=True, slots=True)
class Services:
    """Everything the endpoints need, built by the composition root."""

    authenticator: Authenticator
    checker: TextChecker
    contexts: ContextSource


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

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/check")
    async def check(body: CheckRequest, caller: Annotated[User, Depends(user)]) -> CheckResponse:
        if body.context is not None:
            context = body.context.to_context()
        else:
            context = await _context_or_empty(current().contexts, body.text, body.dictionary)
        result = await current().checker.check(
            body.text, user=caller.name, context=context, dictionary=body.dictionary
        )
        return CheckResponse.of(result, context)

    @app.post("/context")
    async def context(body: ContextRequest, caller: Annotated[User, Depends(user)]) -> ContextBody:
        match await current().contexts.build(body.text, dictionary=body.dictionary):
            case DocumentContext() as built:
                return ContextBody.of(built)
            case Failed(reason=reason):
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=reason)

    return app


async def _context_or_empty(
    contexts: ContextSource, text: str, dictionary: list[str]
) -> DocumentContext:
    """A check goes ahead without context rather than failing when it cannot be built."""
    match await contexts.build(text, dictionary=dictionary):
        case DocumentContext() as built:
            return built
        case Failed():
            return DocumentContext.empty()
