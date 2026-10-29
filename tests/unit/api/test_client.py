import asyncio
import json

import httpx
import pytest

from langcheck.api.client import ServiceClient, ServiceError
from langcheck.api.schemas import CheckResponse, ContextBody

CONTEXT = {"summary": "", "register": "neutral", "dialect": "", "glossary": [], "notes": ""}


def client_for(handler: httpx.MockTransport) -> ServiceClient:
    return ServiceClient(
        httpx.AsyncClient(transport=handler), base_url="http://svc.test/", api_key=None
    )


def test_check_posts_the_text_and_parses_the_reply() -> None:
    seen: list[httpx.Request] = []

    def service(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"suggestions": [], "unchecked": [], "context": CONTEXT})

    reply = asyncio.run(client_for(httpx.MockTransport(service)).check("Hi.", dictionary=["Bilbo"]))

    assert isinstance(reply, CheckResponse)
    assert str(seen[0].url) == "http://svc.test/check"
    assert "Authorization" not in seen[0].headers


def test_context_parses_the_reply() -> None:
    def service(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CONTEXT)

    assert isinstance(
        asyncio.run(client_for(httpx.MockTransport(service)).context("Hi.")), ContextBody
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, json={"detail": "invalid API key"}), "HTTP 401"),
        (httpx.Response(200, json={"nope": 1}), "unexpected reply"),
    ],
)
def test_errors_become_service_errors(response: httpx.Response, message: str) -> None:
    client = client_for(httpx.MockTransport(lambda request: response))

    with pytest.raises(ServiceError, match=message):
        asyncio.run(client.context("Hi."))


def test_unreachable_service() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ServiceError, match=r"cannot reach http://svc\.test/check"):
        asyncio.run(client_for(httpx.MockTransport(refuse)).check("Hi."))


def test_dictionary_calls_encode_the_uri_and_return_the_words() -> None:
    seen: list[httpx.Request] = []

    def service(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"words": ["Bilbo"]})

    client = client_for(httpx.MockTransport(service))

    async def run() -> list[list[str]]:
        return [
            await client.words("file:///a b.md"),
            await client.add_words("file:///a b.md", ["Bilbo"]),
            await client.remove_words("file:///a b.md", ["Shire"]),
        ]

    assert asyncio.run(run()) == [["Bilbo"]] * 3
    assert [r.method for r in seen] == ["GET", "POST", "DELETE"]
    assert seen[0].url.raw_path == b"/documents/file%3A%2F%2F%2Fa%20b.md/dictionary"
    assert seen[0].content == b""
    assert json.loads(seen[2].content) == {"words": ["Shire"]}


def test_check_sends_the_document_uri() -> None:
    seen: list[httpx.Request] = []

    def service(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"suggestions": [], "unchecked": [], "context": CONTEXT})

    asyncio.run(client_for(httpx.MockTransport(service)).check("Hi.", uri="file:///a.md"))

    assert json.loads(seen[0].content)["uri"] == "file:///a.md"
