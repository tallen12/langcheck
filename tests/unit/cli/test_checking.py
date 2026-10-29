import asyncio
import json

import httpx
import pytest

from langcheck.cli.checking import ContextUnavailableError, open_local, open_remote
from langcheck.settings.loading import ClientSettings, load_settings

SETTINGS = load_settings(
    ClientSettings,
    overrides={"llm": {"base_url": "http://llm.test/v1", "retries": 0}, "server": {"api_key": "k"}},
    yaml_data={},
)


def failing_llm(request: httpx.Request) -> httpx.Response:
    return httpx.Response(400, text="bad model")


def test_local_check_goes_ahead_without_context_but_context_fails() -> None:
    transport = httpx.MockTransport(failing_llm)

    async def run() -> None:
        async with open_local(SETTINGS, transport=transport) as checker:
            response = await checker.check("Read teh book.")
            assert response.suggestions == []
            assert response.context.summary == ""
            assert [u.reason.startswith("LLM error") for u in response.unchecked] == [True]
            with pytest.raises(ContextUnavailableError, match="HTTP 400"):
                await checker.context("Read teh book.")

    asyncio.run(run())


def test_remote_client_sends_the_api_key() -> None:
    seen: list[httpx.Request] = []

    def service(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"summary": "", "register": "neutral", "dialect": "", "glossary": [], "notes": ""},
        )

    async def run() -> None:
        async with open_remote(
            SETTINGS, "http://svc.test/", transport=httpx.MockTransport(service)
        ) as client:
            await client.context("Hi.")

    asyncio.run(run())

    assert str(seen[0].url) == "http://svc.test/context"
    assert seen[0].headers["Authorization"] == "Bearer k"
    assert json.loads(seen[0].content) == {"text": "Hi.", "dictionary": []}
