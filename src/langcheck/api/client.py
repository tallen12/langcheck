"""A client for a running ``langcheck serve``, used by the CLI."""

from collections.abc import Sequence

import httpx
from pydantic import BaseModel, ValidationError

from langcheck.api.schemas import CheckRequest, CheckResponse, ContextBody, ContextRequest
from langcheck.llm.openai_chat import auth_headers


class ServiceError(Exception):
    """The service could not be reached or refused the request."""


class ServiceClient:
    """Calls ``/check`` and ``/context`` on the service at ``base_url``."""

    def __init__(self, http: httpx.AsyncClient, *, base_url: str, api_key: str | None) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._headers = auth_headers(api_key)

    async def check(self, text: str, *, dictionary: Sequence[str] = ()) -> CheckResponse:
        """Check ``text``; the service builds the document context."""
        body = CheckRequest(text=text, dictionary=list(dictionary))
        return await self._post("check", body, CheckResponse)

    async def context(self, text: str, *, dictionary: Sequence[str] = ()) -> ContextBody:
        """The document context of ``text``."""
        body = ContextRequest(text=text, dictionary=list(dictionary))
        return await self._post("context", body, ContextBody)

    async def _post[T: BaseModel](self, path: str, body: BaseModel, reply: type[T]) -> T:
        url = f"{self._base_url}/{path}"
        try:
            response = await self._http.post(
                url, json=body.model_dump(mode="json", by_alias=True), headers=self._headers
            )
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
