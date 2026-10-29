import asyncio
import json
from collections.abc import Iterator

import httpx
import pytest

from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.llm.models import ChatMessage, ChatOptions, ClientConfig, LLMError, Role

MESSAGES = (ChatMessage(Role.SYSTEM, "Be brief."), ChatMessage(Role.USER, "Hi"))


def completion(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


class Server:
    """Answers requests from a script of responses (or transport errors) and keeps them."""

    def __init__(self, responses: list[httpx.Response | httpx.TransportError]) -> None:
        self._responses: Iterator[httpx.Response | httpx.TransportError] = iter(responses)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = next(self._responses)
        if isinstance(response, httpx.TransportError):
            raise response
        return response


def make_client(
    server: Server, *, retries: int = 3, api_key: str | None = None
) -> tuple[AsyncOpenAIChatClient, list[float]]:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    config = ClientConfig(
        base_url="http://llm.test/v1/", model="m", api_key=api_key, retries=retries
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return AsyncOpenAIChatClient(http, config, sleep=sleep), waits


def test_posts_the_chat_and_returns_the_first_choice() -> None:
    server = Server([completion("Hello")])
    client, waits = make_client(server, api_key="secret")

    reply = asyncio.run(client.complete(MESSAGES, options=ChatOptions(max_tokens=5)))

    assert reply == "Hello"
    assert waits == []
    request = server.requests[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.content)["max_tokens"] == 5


def test_retries_transient_failures_with_backoff_and_retry_after() -> None:
    server = Server(
        [
            httpx.ConnectError("refused"),
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(503, text="busy"),
            completion("ok"),
        ]
    )
    client, waits = make_client(server)

    assert asyncio.run(client.complete(MESSAGES, options=ChatOptions())) == "ok"
    assert waits == [1.0, 7.0, 4.0]


def test_gives_up_after_the_last_retry() -> None:
    server = Server([httpx.Response(502, text="bad gateway")] * 2)
    client, _ = make_client(server, retries=1)

    with pytest.raises(LLMError, match="gave up after 2 attempts: HTTP 502"):
        asyncio.run(client.complete(MESSAGES, options=ChatOptions()))


def test_client_errors_fail_at_once() -> None:
    server = Server([httpx.Response(400, text="bad request")])
    client, waits = make_client(server)

    with pytest.raises(LLMError, match="HTTP 400: bad request"):
        asyncio.run(client.complete(MESSAGES, options=ChatOptions()))
    assert waits == []


def test_limits_requests_in_flight() -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return completion("ok")

    config = ClientConfig(base_url="http://llm.test/v1", model="m", concurrency=2)
    client = AsyncOpenAIChatClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), config
    )

    async def run() -> list[str]:
        return list(
            await asyncio.gather(
                *(client.complete(MESSAGES, options=ChatOptions()) for _ in range(5))
            )
        )

    assert asyncio.run(run()) == ["ok"] * 5
    assert peak == 2
