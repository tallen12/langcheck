from collections.abc import Callable

import pytest

from langcheck.llm.models import ChatCompletion, ChatOptions, ClientConfig


def test_completion_reads_the_first_choice_and_ignores_extra_fields() -> None:
    reply = ChatCompletion.model_validate_json(
        '{"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi",'
        ' "reasoning": "hmm"}, "finish_reason": "stop"}]}'
    )

    assert reply.text == "hi"


def test_completion_needs_a_choice() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ChatCompletion.model_validate_json('{"choices": []}')


URL = "http://x/v1"


@pytest.mark.parametrize(
    "build",
    [
        lambda: ClientConfig(base_url="", model="m"),
        lambda: ClientConfig(base_url=URL, model=""),
        lambda: ClientConfig(base_url=URL, model="m", timeout=0),
        lambda: ClientConfig(base_url=URL, model="m", retries=-1),
        lambda: ClientConfig(base_url=URL, model="m", backoff=-1),
        lambda: ClientConfig(base_url=URL, model="m", concurrency=0),
        lambda: ChatOptions(temperature=-0.1),
        lambda: ChatOptions(max_tokens=0),
    ],
)
def test_settings_are_validated(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()
