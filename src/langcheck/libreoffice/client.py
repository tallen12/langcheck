"""The langcheck service over ``urllib``, for LibreOffice's Python."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote

from langcheck.libreoffice.schema import (
    DocumentState,
    SchemaError,
    parse_document,
    parse_version,
)

DEFAULT_TIMEOUT = 30.0

# typing.Callable: this alias is evaluated at run time, on Python 3.8 too.
Opener = Callable[[urllib.request.Request, float], bytes]
"""Sends a request with a timeout and returns the body; raises ``OSError`` on failure."""


class ServiceError(Exception):
    """The service could not be reached or answered with an error."""


@dataclass(frozen=True)
class Connection:
    """Where the service is. ``api_key`` is left out of ``repr``."""

    url: str
    api_key: str = field(default="", repr=False)


def urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    """The real :data:`Opener`."""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


class ServiceClient:
    """``PUT /documents/{uri}`` and ``GET /documents/{uri}/suggestions``."""

    def __init__(
        self,
        connection: Connection,
        *,
        opener: Opener = urlopen,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._connection = connection
        self._opener = opener
        self._timeout = timeout

    def put_document(self, uri: str, text: str) -> int:
        """Open or update the document; the service's version."""
        body = json.dumps({"text": text}).encode("utf-8")
        reply = self._call("PUT", _document_path(uri), body)
        try:
            return parse_version(reply)
        except SchemaError as error:
            raise _unexpected(uri, error) from error

    def document(self, uri: str) -> DocumentState:
        """The current version's suggestions and check state."""
        reply = self._call("GET", f"{_document_path(uri)}/suggestions", None)
        try:
            return parse_document(reply)
        except SchemaError as error:
            raise _unexpected(uri, error) from error

    def _call(self, method: str, path: str, body: bytes | None) -> object:
        url = f"{self._connection.url.rstrip('/')}/{path}"
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._connection.api_key:
            headers["Authorization"] = f"Bearer {self._connection.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            raw = self._opener(request, self._timeout)
        except urllib.error.HTTPError as error:
            msg = f"{url} answered HTTP {error.code}"
            raise ServiceError(msg) from error
        except OSError as error:
            msg = f"cannot reach {url}: {error}"
            raise ServiceError(msg) from error
        try:
            data: object = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            msg = f"{url} sent invalid JSON"
            raise ServiceError(msg) from error
        return data


def _unexpected(uri: str, error: SchemaError) -> ServiceError:
    # A malformed reply is a service failure like any other: the worker logs it and
    # carries on, where a stray SchemaError would end its thread for good.
    return ServiceError(f"unexpected reply for {uri}: {error}")


def _document_path(uri: str) -> str:
    return f"documents/{quote(uri, safe='')}"
