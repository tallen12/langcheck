"""The ``/documents/{uri}`` endpoints: open documents, their suggestions and events.

``{uri}`` is the client's document URI, URL-encoded; it may contain slashes once decoded.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from langcheck.api.schemas import (
    AcceptBody,
    AcceptRequest,
    ConflictBody,
    DocumentBody,
    RenameRequest,
    UpdateRequest,
    VersionBody,
    WordsBody,
    WordsRequest,
)
from langcheck.auth.models import User
from langcheck.dictionary.protocols import DictionaryStore
from langcheck.dictionary.words import InvalidWordError
from langcheck.documents.events import Closed, DocumentEvent, Snapshot
from langcheck.documents.models import DocumentKey
from langcheck.documents.protocols import DocumentService
from langcheck.documents.sessions import Accepted, NotOpen, Updated, VersionConflict

KEEPALIVE_SECONDS = 15.0
NOT_OPEN = "document is not open"

type Authenticate = Callable[..., Coroutine[object, object, User]]


def document_router(
    documents: Callable[[], DocumentService],
    dictionary: Callable[[], DictionaryStore],
    authenticate: Authenticate,
    *,
    keepalive: float = KEEPALIVE_SECONDS,
) -> APIRouter:
    """The router; ``documents`` and ``dictionary`` return the running services,
    ``authenticate`` is the dependency that identifies the caller.
    """
    router = APIRouter(prefix="/documents")
    Caller = Annotated[User, Depends(authenticate)]

    # Registered before the document routes: "/{uri:path}" would also match these.
    @router.get("/{uri:path}/dictionary")
    async def words(uri: str, caller: Caller) -> WordsBody:
        found = await asyncio.to_thread(dictionary().words, caller.name, uri)
        return WordsBody(words=sorted(found))

    @router.post("/{uri:path}/dictionary")
    async def add_words(uri: str, body: WordsRequest, caller: Caller) -> WordsBody:
        try:
            await asyncio.to_thread(dictionary().add, caller.name, uri, body.words)
        except InvalidWordError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        return await _dictionary_changed(uri, caller)

    @router.delete("/{uri:path}/dictionary")
    async def remove_words(uri: str, body: WordsRequest, caller: Caller) -> WordsBody:
        await asyncio.to_thread(dictionary().remove, caller.name, uri, body.words)
        return await _dictionary_changed(uri, caller)

    async def _dictionary_changed(uri: str, caller: User) -> WordsBody:
        await documents().dictionary_changed(DocumentKey(caller.name, uri))
        return await words(uri, caller)

    @router.put("/{uri:path}", responses={409: {"model": ConflictBody}})
    async def update(uri: str, body: UpdateRequest, caller: Caller) -> VersionBody:
        key = DocumentKey(caller.name, uri)
        match await documents().update(key, body.text, base_version=body.base_version):
            case Updated(version=version):
                return VersionBody(version=version)
            case VersionConflict(current=current):
                raise _conflict(current)

    @router.delete("/{uri:path}", status_code=status.HTTP_204_NO_CONTENT)
    async def close(uri: str, caller: Caller) -> Response:
        if not await documents().close(DocumentKey(caller.name, uri)):
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_OPEN)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post("/{uri:path}/rename", responses={409: {"model": ConflictBody}})
    async def rename(uri: str, body: RenameRequest, caller: Caller) -> VersionBody:
        match await documents().rename(DocumentKey(caller.name, uri), body.uri):
            case Updated(version=version):
                return VersionBody(version=version)
            case VersionConflict(current=current):
                raise _conflict(current, detail="a document is already open at that URI")
            case NotOpen():
                raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_OPEN)

    @router.get("/{uri:path}/suggestions", responses={409: {"model": ConflictBody}})
    async def suggestions(uri: str, caller: Caller, version: int | None = None) -> DocumentBody:
        document = await documents().get(DocumentKey(caller.name, uri))
        if document is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_OPEN)
        if version is not None and version != document.version:
            raise _conflict(document.version)
        return DocumentBody.of(Snapshot.of(document))

    @router.post(
        "/{uri:path}/suggestions/{suggestion_id}/accept", responses={409: {"model": ConflictBody}}
    )
    async def accept(
        uri: str, suggestion_id: str, body: AcceptRequest, caller: Caller
    ) -> AcceptBody:
        key = DocumentKey(caller.name, uri)
        result = await documents().accept(
            key, suggestion_id, option=body.option, base_version=body.base_version
        )
        match result:
            case Accepted(version=version, span=span, replacement=replacement):
                return AcceptBody(
                    version=version, start=span.start, end=span.end, replacement=replacement
                )
            case VersionConflict(current=current):
                raise _conflict(current)
            case NotOpen():
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such suggestion or option")

    @router.post(
        "/{uri:path}/suggestions/{suggestion_id}/dismiss",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def dismiss(uri: str, suggestion_id: str, caller: Caller) -> Response:
        if not await documents().dismiss(DocumentKey(caller.name, uri), suggestion_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such suggestion")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/{uri:path}/events")
    async def events(uri: str, caller: Caller) -> StreamingResponse:
        key = DocumentKey(caller.name, uri)
        if await documents().get(key) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_OPEN)
        return StreamingResponse(
            _stream(documents(), key, keepalive=keepalive),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _conflict(current: int, *, detail: str = "version conflict") -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, {"message": detail, "current": current})


async def _stream(
    documents: DocumentService, key: DocumentKey, *, keepalive: float
) -> AsyncGenerator[str]:
    async with documents.watch(key) as queue:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=keepalive)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield sse(event)
            if isinstance(event, Closed):
                return


def sse(event: DocumentEvent) -> str:
    """One Server-Sent Event: ``snapshot`` with a :class:`DocumentBody`, or ``closed``."""
    match event:
        case Snapshot():
            return f"event: snapshot\ndata: {DocumentBody.of(event).model_dump_json()}\n\n"
        case Closed():
            return "event: closed\ndata: {}\n\n"
