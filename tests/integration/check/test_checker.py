"""Checks text against a real OpenAI-compatible server named by
``LANGCHECK_TEST_LLM_BASE_URL``.

Run with ``LANGCHECK_TEST_LLM_BASE_URL=http://localhost:11434/v1 uv run pytest -m llm``.
``LANGCHECK_TEST_LLM_MODEL`` picks the model (default ``qwen3:8b``),
``LANGCHECK_TEST_LLM_API_KEY`` the key, and ``LANGCHECK_TEST_LLM_REASONING_SWITCH=1``
turns reasoning off on servers that take the switch (Ollama does).
"""

import asyncio
import os

import httpx
import pytest

from langcheck.check.checker import Checker, CheckResult
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.resolve import ValidationPolicy
from langcheck.context.builder import ContextBuilder
from langcheck.context.models import DocumentContext
from langcheck.llm.async_openai_chat import AsyncOpenAIChatClient
from langcheck.llm.models import ChatOptions, ClientConfig
from langcheck.text.regex_splitter import RegexSentenceSplitter

pytestmark = pytest.mark.llm

TEXT = (
    "Bilbo Baggins lived in a hobbit-hole under the Hill.\n\n"
    "Every morning he would walk to teh market and buy fresh bread."
)


@pytest.fixture
def config() -> ClientConfig:
    base_url = os.environ.get("LANGCHECK_TEST_LLM_BASE_URL", "")
    if not base_url:
        pytest.skip("LANGCHECK_TEST_LLM_BASE_URL is not set")
    key = os.environ.get("LANGCHECK_TEST_LLM_API_KEY", "")
    return ClientConfig(
        base_url=base_url,
        model=os.environ.get("LANGCHECK_TEST_LLM_MODEL", "qwen3:8b"),
        api_key=key if key else None,
        retries=1,
        reasoning_switch=os.environ.get("LANGCHECK_TEST_LLM_REASONING_SWITCH", "") == "1",
    )


async def build_and_check(config: ClientConfig) -> tuple[DocumentContext, CheckResult]:
    async with httpx.AsyncClient() as http:
        llm = AsyncOpenAIChatClient(http, config)
        options = ChatOptions()
        built = await ContextBuilder(llm, options=options).build(TEXT)
        assert isinstance(built, DocumentContext), built
        checker = Checker(
            llm,
            splitter=RegexSentenceSplitter(),
            cache=MemoryResultCache(),
            model=config.model,
            options=options,
            policy=ValidationPolicy(),
        )
        return built, await checker.check(TEXT, user="test", context=built)


def test_builds_a_context_and_finds_a_misspelling(config: ClientConfig) -> None:
    context, result = asyncio.run(build_and_check(config))

    assert context.summary
    assert result.unchecked == ()
    fixes = {s.original: [o.replacement for o in s.options] for s in result.suggestions}
    assert "the" in fixes.get("teh", [])
    assert "Bilbo" not in fixes
