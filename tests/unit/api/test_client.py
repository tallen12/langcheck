import asyncio

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
