"""``AsyncLLMClient`` over an OpenAI-compatible ``/chat/completions`` API, for the service.

Same request, retry and error rules as :class:`langcheck.llm.openai_chat.OpenAIChatClient`,
on ``httpx.AsyncClient`` so checks never block the event loop.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence

import httpx

from langcheck.llm.models import ChatMessage, ChatOptions, ClientConfig, LLMError
from langcheck.llm.openai_chat import (
    COMPLETIONS_PATH,
    auth_headers,
    backoff_delay,
    describe,
    reply_text,
    request_body,
    retry_after_seconds,
    retryable,
)


class AsyncOpenAIChatClient:
    """Sends each chat to ``{base_url}/chat/completions`` and returns the first choice.

    A semaphore keeps at most ``config.concurrency`` requests in flight across tasks; a
    task waiting out a backoff does not hold a slot.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        config: ClientConfig,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._config = config
        self._sleep = sleep
        self._slots = asyncio.Semaphore(config.concurrency)
        self._url = f"{config.base_url.rstrip('/')}/{COMPLETIONS_PATH}"

    async def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """The model's reply text to ``messages``; raises ``LLMError`` when retries run out."""
        body = request_body(
            self._config.model,
            messages,
            options,
            reasoning_switch=self._config.reasoning_switch,
        )
        headers = auth_headers(self._config.api_key)
        last_problem = "no attempt made"
        retry_after: float | None = None
        for attempt in range(self._config.retries + 1):
            if attempt:
                await self._sleep(backoff_delay(self._config, attempt - 1, retry_after))
            async with self._slots:
                try:
                    response = await self._http.post(
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
