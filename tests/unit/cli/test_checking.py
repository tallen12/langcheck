import asyncio
import json
from pathlib import Path

import httpx
import pytest

from langcheck.cli.checking import ContextUnavailableError, open_local, open_remote
from langcheck.settings.loading import ClientSettings, load_settings
from langcheck.tagging.loaded import Trained
from tests.conftest import WordTableTagger

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
            json={"summary": "", "register": "neutral", "dialect": "", "notes": ""},
        )

    async def run() -> None:
        async with open_remote(
            SETTINGS, "http://svc.test/", transport=httpx.MockTransport(service)
        ) as client:
            await client.context("Hi.")

    asyncio.run(run())

    assert str(seen[0].url) == "http://svc.test/context"
    assert seen[0].headers["Authorization"] == "Bearer k"
    assert json.loads(seen[0].content) == {"text": "Hi."}


def test_a_configured_tagger_checks_locally() -> None:
    settings = load_settings(
        ClientSettings,
        overrides={
            "llm": {"base_url": "http://llm.test/v1", "retries": 0},
            "correction": {"checkpoint": "tagger"},
        },
        yaml_data={},
    )

    def llm(request: httpx.Request) -> httpx.Response:
        system = json.loads(request.content)["messages"][0]["content"]
        if "error detector" not in system:
            return httpx.Response(400, text="no context")
        content = json.dumps(
            {
                "flags": [
                    {
                        "flag": "F1",
                        "corrections": [
                            {"replacement": "goes", "kind": "VerbForm", "explanation": "Agrees."}
                        ],
                    }
                ]
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    def load(path: Path, *, device: str) -> Trained:
        return Trained(WordTableTagger({"go": 0.9}), 0.5, "base", 0.5)

    async def run() -> list[str]:
        transport = httpx.MockTransport(llm)
        async with open_local(settings, transport=transport, load_tagger=load) as checker:
            response = await checker.check("He go home.")
            return [s.options[0].replacement for s in response.suggestions]

    assert asyncio.run(run()) == ["goes"]
