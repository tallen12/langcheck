"""A client for a running ``langcheck serve``, used by the CLI."""

from collections.abc import AsyncIterator, Sequence
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from langcheck.api.schemas import (
    CheckRequest,
    CheckResponse,
    ContextBody,
    ContextRequest,
    DocumentBody,
    UpdateRequest,
    VersionBody,
    WordsBody,
    WordsRequest,
)
from langcheck.llm.openai_chat import auth_headers


class ServiceError(Exception):
    """The service could not be reached or refused the request."""


class ServiceClient:
    """Calls ``/check``, ``/context`` and the ``/documents`` endpoints of the service."""

    def __init__(self, http: httpx.AsyncClient, *, base_url: str, api_key: str | None) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._headers = auth_headers(api_key)

    async def check(
        self, text: str, *, dictionary: Sequence[str] = (), uri: str | None = None
    ) -> CheckResponse:
        """Check ``text``; the service builds the document context. With ``uri``, that
        document's stored dictionary applies too.
        """
        body = CheckRequest(text=text, dictionary=list(dictionary), uri=uri)
        return await self._send("POST", "check", body, CheckResponse)

    async def context(self, text: str, *, dictionary: Sequence[str] = ()) -> ContextBody:
        """The document context of ``text``."""
        body = ContextRequest(text=text, dictionary=list(dictionary))
        return await self._send("POST", "context", body, ContextBody)

    async def words(self, uri: str) -> list[str]:
        """The dictionary of the document at ``uri``, sorted."""
        return (await self._send("GET", _dictionary_path(uri), None, WordsBody)).words

    async def add_words(self, uri: str, words: Sequence[str]) -> list[str]:
        """Add ``words`` to the document's dictionary; the dictionary after."""
        body = WordsRequest(words=list(words))
        return (await self._send("POST", _dictionary_path(uri), body, WordsBody)).words

    async def remove_words(self, uri: str, words: Sequence[str]) -> list[str]:
        """Remove ``words`` from the document's dictionary; the dictionary after."""
        body = WordsRequest(words=list(words))
        return (await self._send("DELETE", _dictionary_path(uri), body, WordsBody)).words

    async def put_document(self, uri: str, text: str) -> int:
        """Open or update the document with its full text; the server's new version."""
        body = UpdateRequest(text=text)
        return (await self._send("PUT", _document_path(uri), body, VersionBody)).version

    async def close_document(self, uri: str) -> bool:
        """Close the document; ``False`` if it was not open."""
        response = await self._request("DELETE", _document_path(uri), None, missing_ok=True)
        return response.status_code != httpx.codes.NOT_FOUND

    async def dismiss(self, uri: str, suggestion_id: str) -> bool:
        """Dismiss a suggestion; ``False`` if it is gone already."""
        path = f"{_document_path(uri)}/suggestions/{quote(suggestion_id, safe='')}/dismiss"
        response = await self._request("POST", path, None, missing_ok=True)
        return response.status_code != httpx.codes.NOT_FOUND

    async def events(self, uri: str) -> AsyncIterator[DocumentBody | None]:
        """The document's snapshots as Server-Sent Events arrive; ``None`` means the
        document closed, after which the stream ends. Keep-alive comments are skipped.
        """
        url = f"{self._base_url}/{_document_path(uri)}/events"
        try:
            async with self._http.stream(
                "GET", url, headers=self._headers, timeout=None
            ) as response:
                if response.is_error:
                    await response.aread()
                    msg = f"{url} answered HTTP {response.status_code}: {response.text[:200]}"
                    raise ServiceError(msg)
                async for event, data in parse_sse(response.aiter_lines()):
                    if event == "closed":
                        yield None
                        return
                    if event == "snapshot":
                        yield _parse(url, data, DocumentBody)
        except httpx.TransportError as error:
            msg = f"lost {url}: {error}"
            raise ServiceError(msg) from error

    async def _send[T: BaseModel](
        self, method: str, path: str, body: BaseModel | None, reply: type[T]
    ) -> T:
        response = await self._request(method, path, body, missing_ok=False)
        return _parse(str(response.url), response.text, reply)

    async def _request(
        self, method: str, path: str, body: BaseModel | None, *, missing_ok: bool
    ) -> httpx.Response:
        url = f"{self._base_url}/{path}"
        payload = body.model_dump(mode="json", by_alias=True) if body is not None else None
        try:
            response = await self._http.request(method, url, json=payload, headers=self._headers)
        except httpx.TransportError as error:
            msg = f"cannot reach {url}: {error}"
            raise ServiceError(msg) from error
        missing = missing_ok and response.status_code == httpx.codes.NOT_FOUND
        if response.is_error and not missing:
            msg = f"{url} answered HTTP {response.status_code}: {response.text[:200]}"
            raise ServiceError(msg)
        return response


async def parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, str]]:
    """``(event, data)`` pairs from Server-Sent Event lines; comments are skipped and
    multi-line data is joined with newlines, as the SSE format defines.
    """
    event = "message"
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield event, "\n".join(data)
            event = "message"
            data = []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)


def _parse[T: BaseModel](url: str, text: str, reply: type[T]) -> T:
    try:
        return reply.model_validate_json(text)
    except ValidationError as error:
        msg = f"unexpected reply from {url}: {text[:200]}"
        raise ServiceError(msg) from error


def _document_path(uri: str) -> str:
    return f"documents/{quote(uri, safe='')}"


def _dictionary_path(uri: str) -> str:
    return f"{_document_path(uri)}/dictionary"
