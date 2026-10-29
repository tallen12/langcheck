import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from langcheck.api.client import ServiceClient, ServiceError, parse_sse
from langcheck.api.schemas import CheckResponse, ContextBody, DocumentBody

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


def test_document_calls_use_the_encoded_uri() -> None:
    seen: list[httpx.Request] = []

    def service(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "PUT":
            return httpx.Response(200, json={"version": 3})
        status = 404 if "gone" in str(request.url) else 204
        return httpx.Response(status)

    client = client_for(httpx.MockTransport(service))

    async def run() -> tuple[int, bool, bool, bool]:
        return (
            await client.put_document("file:///a.md", "Hi."),
            await client.close_document("file:///a.md"),
            await client.close_document("file:///gone.md"),
            await client.dismiss("file:///a.md", "sg_1"),
        )

    assert asyncio.run(run()) == (3, True, False, True)
    assert seen[0].url.raw_path == b"/documents/file%3A%2F%2F%2Fa.md"
    assert json.loads(seen[0].content) == {"text": "Hi.", "base_version": None}
    assert seen[3].url.raw_path == b"/documents/file%3A%2F%2F%2Fa.md/suggestions/sg_1/dismiss"


def test_document_errors_other_than_missing_raise() -> None:
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(500, text="boom")))

    with pytest.raises(ServiceError, match="HTTP 500"):
        asyncio.run(client.close_document("file:///a.md"))


SNAPSHOT = {"version": 2, "state": "done", "suggestions": [], "unchecked": []}


def test_events_yield_snapshots_then_none_on_close() -> None:
    stream = (
        ": keepalive\n\n"
        f"event: snapshot\ndata: {json.dumps(SNAPSHOT)}\n\n"
        "event: closed\ndata: {}\n\n"
        f"event: snapshot\ndata: {json.dumps(SNAPSHOT)}\n\n"
    )
    client = client_for(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=stream, headers={"content-type": "text/event-stream"}
            )
        )
    )

    async def run() -> list[DocumentBody | None]:
        return [event async for event in client.events("file:///a.md")]

    events = asyncio.run(run())

    assert [e.version if e is not None else None for e in events] == [2, None]


def test_events_report_errors() -> None:
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(404, text="nope")))

    async def run() -> None:
        async for _ in client.events("file:///a.md"):
            pass

    with pytest.raises(ServiceError, match="HTTP 404"):
        asyncio.run(run())


def test_parse_sse_joins_data_lines_and_skips_comments() -> None:
    async def lines() -> AsyncIterator[str]:
        for line in ["event: a", "data: one", "data:two", ": note", "", "data: x", "", "", "id: 7"]:
            yield line

    async def run() -> list[tuple[str, str]]:
        return [pair async for pair in parse_sse(lines())]

    assert asyncio.run(run()) == [("a", "one\ntwo"), ("message", "x")]
