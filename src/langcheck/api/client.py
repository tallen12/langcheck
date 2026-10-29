"""A client for a running ``langcheck serve``, used by the CLI."""

from collections.abc import Sequence
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from langcheck.api.schemas import (
    CheckRequest,
    CheckResponse,
    ContextBody,
    ContextRequest,
    WordsBody,
    WordsRequest,
)
from langcheck.llm.openai_chat import auth_headers


class ServiceError(Exception):
    """The service could not be reached or refused the request."""


class ServiceClient:
    """Calls ``/check``, ``/context`` and document dictionaries on the service."""

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

    async def _send[T: BaseModel](
        self, method: str, path: str, body: BaseModel | None, reply: type[T]
    ) -> T:
        url = f"{self._base_url}/{path}"
        payload = body.model_dump(mode="json", by_alias=True) if body is not None else None
        try:
            response = await self._http.request(method, url, json=payload, headers=self._headers)
        except httpx.TransportError as error:
            msg = f"cannot reach {url}: {error}"
            raise ServiceError(msg) from error
        if response.is_error:
            msg = f"{url} answered HTTP {response.status_code}: {response.text[:200]}"
            raise ServiceError(msg)
        try:
            return reply.model_validate_json(response.content)
        except ValidationError as error:
            msg = f"unexpected reply from {url}: {response.text[:200]}"
            raise ServiceError(msg) from error


def _dictionary_path(uri: str) -> str:
    return f"documents/{quote(uri, safe='')}/dictionary"
