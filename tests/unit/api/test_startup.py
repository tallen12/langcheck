import json
from datetime import UTC
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from langcheck.api.startup import create_app, utc_now
from langcheck.auth.sql_store import SqlApiKeyStore
from langcheck.auth.tokens import add_user
from langcheck.database.schema import open_database
from langcheck.settings.loading import ServerSettings, load_settings


def settings_for(tmp_path: Path, *, auth: str) -> ServerSettings:
    return load_settings(
        ServerSettings,
        overrides={
            "auth": {"mode": auth},
            "database": {"url": f"sqlite:///{tmp_path / 'langcheck.db'}"},
            "llm": {"base_url": "http://llm.test/v1", "retries": 0},
        },
        yaml_data={},
    )


def fake_llm(request: httpx.Request) -> httpx.Response:
    """An LLM server that describes every document the same way and finds no errors."""
    system = json.loads(request.content)["messages"][0]["content"]
    if "proofreader" in system and "numbered sentences" in system:
        content = json.dumps({"issues": []})
    else:
        content = json.dumps(
            {
                "summary": "A note.",
                "register": "neutral",
                "dialect": "en-US",
                "glossary": [],
                "notes": "",
            }
        )
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_utc_now_is_aware() -> None:
    assert utc_now().tzinfo is UTC


def test_none_mode_checks_end_to_end_against_a_fake_llm(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path, auth="none"), transport=httpx.MockTransport(fake_llm))

    with TestClient(app) as client:
        response = client.post("/check", json={"text": "All is well."})

    assert response.status_code == 200
    assert response.json()["suggestions"] == []
    assert response.json()["context"]["summary"] == "A note."


def test_api_key_mode_accepts_keys_from_the_database(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, auth="api_key")
    store = SqlApiKeyStore(open_database(settings.database.url), clock=utc_now)
    token = add_user(store, "ana")
    app = create_app(settings, transport=httpx.MockTransport(fake_llm))

    with TestClient(app) as client:
        refused = client.post("/context", json={"text": "Hi."})
        accepted = client.post(
            "/context", json={"text": "Hi."}, headers={"Authorization": f"Bearer {token}"}
        )

    assert refused.status_code == 401
    assert accepted.status_code == 200
