"""``LLMClient`` over an OpenAI-compatible ``/chat/completions`` API (Ollama, llama.cpp, vLLM).

Tested against Ollama 0.24, whose OpenAI-compatible endpoint ignores Ollama's own
``think`` flag but honours ``reasoning_effort: "none"``; that field is only sent when
``ClientConfig.disable_reasoning`` is set.
"""

import math
import threading
import time
from collections.abc import Callable, Sequence

import httpx
from pydantic import ValidationError

from langcheck.llm.models import ChatCompletion, ChatMessage, ChatOptions, ClientConfig, LLMError

COMPLETIONS_PATH = "chat/completions"
RETRY_AFTER = "Retry-After"
TOO_MANY_REQUESTS = 429
# 501, 505 and other 5xx never succeed on a retry.
TRANSIENT_STATUSES = frozenset({TOO_MANY_REQUESTS, 500, 502, 503, 504})


class OpenAIChatClient:
    """Sends each chat to ``{base_url}/chat/completions`` and returns the first choice.

    Retries connection errors, timeouts, 429, 500, 502, 503 and 504 with exponential backoff (or
    the server's ``Retry-After``); other 4xx replies fail at once. A semaphore keeps at
    most ``config.concurrency`` requests in flight across threads.
    """

    def __init__(
        self,
        http: httpx.Client,
        config: ClientConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._config = config
        self._sleep = sleep
        self._slots = threading.BoundedSemaphore(config.concurrency)
        self._url = f"{config.base_url.rstrip('/')}/{COMPLETIONS_PATH}"

    def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """The model's reply text to ``messages``; raises ``LLMError`` when retries run out."""
        body = request_body(
            self._config.model,
            messages,
            options,
            disable_reasoning=self._config.disable_reasoning,
        )
        headers = auth_headers(self._config.api_key)
        last_problem = "no attempt made"
        retry_after: float | None = None
        for attempt in range(self._config.retries + 1):
            if attempt:
                self._sleep(backoff_delay(self._config, attempt - 1, retry_after))
            with self._slots:
                try:
                    response = self._http.post(
                        self._url, json=body, headers=headers, timeout=self._config.timeout
                    )
                except httpx.TransportError as error:
                    last_problem = f"{type(error).__name__}: {error}"
                    retry_after = None
                    continue
            if retryable(response.status_code):
                last_problem = describe(response)
                retry_after = retry_after_seconds(response)
                continue
            if response.is_error:
                raise LLMError(describe(response))
            return reply_text(response)
        raise LLMError(f"gave up after {self._config.retries + 1} attempts: {last_problem}")


def request_body(
    model: str,
    messages: Sequence[ChatMessage],
    options: ChatOptions,
    *,
    disable_reasoning: bool = False,
) -> dict[str, object]:
    """The JSON body of a ``/chat/completions`` request.

    ``reasoning_effort: "none"`` is added only with ``disable_reasoning``, for servers
    known to take it.
    """
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "temperature": options.temperature,
        "stream": False,
    }
    if options.max_tokens is not None:
        body["max_tokens"] = options.max_tokens
    if options.json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": options.schema_name, "schema": dict(options.json_schema)},
        }
    if disable_reasoning and not options.think:
        body["reasoning_effort"] = "none"
    return body


def auth_headers(api_key: str | None) -> dict[str, str]:
    """``Authorization: Bearer`` when a key is set, else nothing."""
    return {} if api_key is None or api_key == "" else {"Authorization": f"Bearer {api_key}"}


def backoff_delay(config: ClientConfig, retry: int, retry_after: float | None) -> float:
    """Seconds to wait before retry number ``retry`` (0-based), capped at ``max_backoff``."""
    wait = config.backoff * 2.0**retry if retry_after is None else retry_after
    return min(wait, config.max_backoff)


def retryable(status: int) -> bool:
    """Whether a reply with ``status`` may succeed when sent again."""
    return status in TRANSIENT_STATUSES


def retry_after_seconds(response: httpx.Response) -> float | None:
    """The server's ``Retry-After`` in seconds, when it sent a usable one."""
    value = response.headers.get(RETRY_AFTER)
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    # "nan" parses as a float but cannot be slept; "inf" is capped by max_backoff.
    return max(seconds, 0.0) if not math.isnan(seconds) else None


def describe(response: httpx.Response) -> str:
    """A short description of a failed reply for error messages."""
    return f"HTTP {response.status_code}: {response.text[:200]}"


def reply_text(response: httpx.Response) -> str:
    """The first choice's text of a successful reply; ``LLMError`` if it is malformed."""
    try:
        return ChatCompletion.model_validate_json(response.content).text
    except ValidationError as error:
        raise LLMError(f"unexpected reply: {response.text[:200]}") from error
