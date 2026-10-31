import json
from collections.abc import Callable, Iterator

import httpx
import pytest

from langcheck.llm.models import ChatMessage, ChatOptions, ClientConfig, LLMError, Role
from langcheck.llm.openai_chat import OpenAIChatClient, auth_headers, request_body

MESSAGES = (ChatMessage(Role.SYSTEM, "Be brief."), ChatMessage(Role.USER, "Hi"))
BASE_URL = "http://llm.test/v1"


def completion(text: str | None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )


class Server:
    """Answers requests from a script of responses (or transport errors) and keeps them."""

    def __init__(self, responses: list[httpx.Response | httpx.TransportError]) -> None:
        self._responses: Iterator[httpx.Response | httpx.TransportError] = iter(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = next(self._responses)
        if isinstance(response, httpx.TransportError):
            raise response
        return response


def make_client(
    server: Server,
    *,
    api_key: str | None = None,
    retries: int = 3,
    disable_reasoning: bool = False,
) -> tuple[OpenAIChatClient, list[float]]:
    waits: list[float] = []
    config = ClientConfig(
        base_url=BASE_URL,
        model="m",
        api_key=api_key,
        retries=retries,
        backoff=1.0,
        disable_reasoning=disable_reasoning,
    )
    http = httpx.Client(transport=httpx.MockTransport(server))
    return OpenAIChatClient(http, config, sleep=waits.append), waits


def test_posts_the_chat_and_returns_the_first_choice() -> None:
    server = Server([completion("Hello")])
    client, waits = make_client(server)

    reply = client.complete(MESSAGES, options=ChatOptions(temperature=0.2, max_tokens=50))

    assert reply == "Hello"
    assert waits == []
    request = server.requests[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "m"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 50
    assert "reasoning_effort" not in body


def test_turns_reasoning_off_only_on_servers_that_take_the_switch() -> None:
    server = Server([completion("a")])
    client, _ = make_client(server, disable_reasoning=True)

    client.complete(MESSAGES, options=ChatOptions())

    assert json.loads(server.requests[0].content)["reasoning_effort"] == "none"


@pytest.mark.parametrize("status", [501, 505])
def test_permanent_server_errors_fail_at_once(status: int) -> None:
    server = Server([httpx.Response(status, text="no")])
    client, waits = make_client(server)

    with pytest.raises(LLMError, match=f"HTTP {status}"):
        client.complete(MESSAGES, options=ChatOptions())
    assert waits == []


def test_sends_the_bearer_token_only_when_a_key_is_set() -> None:
    with_key = Server([completion("a")])
    without_key = Server([completion("b")])

    make_client(with_key, api_key="secret")[0].complete(MESSAGES, options=ChatOptions())
    make_client(without_key)[0].complete(MESSAGES, options=ChatOptions())

    assert with_key.requests[0].headers["Authorization"] == "Bearer secret"
    assert "Authorization" not in without_key.requests[0].headers


def test_auth_headers_skip_an_empty_key() -> None:
    assert auth_headers("") == {}
    assert auth_headers(None) == {}


def test_config_repr_hides_the_key() -> None:
    config = ClientConfig(base_url=BASE_URL, model="m", api_key="secret")

    assert "secret" not in repr(config)


@pytest.mark.parametrize(
    "failure",
    [
        lambda: httpx.Response(503, text="busy"),
        lambda: httpx.Response(500, text="boom"),
        lambda: httpx.ConnectError("refused"),
        lambda: httpx.ReadTimeout("slow"),
    ],
)
def test_retries_server_and_connection_errors_with_backoff(
    failure: Callable[[], httpx.Response | httpx.TransportError],
) -> None:
    server = Server([failure(), failure(), completion("ok")])
    client, waits = make_client(server)

    reply = client.complete(MESSAGES, options=ChatOptions())

    assert reply == "ok"
    assert waits == [1.0, 2.0]


def test_honours_retry_after_on_429() -> None:
    server = Server([httpx.Response(429, headers={"Retry-After": "7"}), completion("ok")])
    client, waits = make_client(server)

    assert client.complete(MESSAGES, options=ChatOptions()) == "ok"
    assert waits == [7.0]


def test_ignores_a_retry_after_that_is_not_a_number() -> None:
    server = Server([httpx.Response(429, headers={"Retry-After": "nan"}), completion("ok")])
    client, waits = make_client(server)

    assert client.complete(MESSAGES, options=ChatOptions()) == "ok"
    assert waits == [1.0]


def test_gives_up_after_the_last_retry() -> None:
    server = Server([httpx.Response(502, text="bad gateway")] * 3)
    client, waits = make_client(server, retries=2)

    with pytest.raises(LLMError, match="gave up after 3 attempts: HTTP 502"):
        client.complete(MESSAGES, options=ChatOptions())
    assert len(server.requests) == 3
    assert waits == [1.0, 2.0]


def test_client_errors_fail_at_once() -> None:
    server = Server([httpx.Response(404, text="model not found")])
    client, waits = make_client(server)

    with pytest.raises(LLMError, match="HTTP 404: model not found"):
        client.complete(MESSAGES, options=ChatOptions())
    assert waits == []


def test_rejects_a_reply_that_is_not_a_completion() -> None:
    server = Server([httpx.Response(200, json={"error": "nope"})])
    client, _ = make_client(server)

    with pytest.raises(LLMError, match="unexpected reply"):
        client.complete(MESSAGES, options=ChatOptions())


def test_missing_content_reads_as_empty() -> None:
    server = Server([completion(None)])
    client, _ = make_client(server)

    assert client.complete(MESSAGES, options=ChatOptions()) == ""


def test_request_body_adds_the_schema_and_keeps_thinking_when_asked() -> None:
    schema = {"type": "object"}

    body = request_body(
        "m",
        MESSAGES,
        ChatOptions(json_schema=schema, schema_name="s", think=True),
        disable_reasoning=True,
    )

    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "s", "schema": {"type": "object"}},
    }
    assert "reasoning_effort" not in body
    assert "max_tokens" not in body
