import json
import time
from datetime import UTC
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from langcheck.api.app import build_app
from langcheck.api.startup import create_app, open_services, utc_now
from langcheck.auth.sql_store import SqlApiKeyStore
from langcheck.auth.tokens import add_user
from langcheck.database.schema import open_database
from langcheck.scoring.factory import ScorerAdapter, ScorerConfig
from langcheck.scoring.protocols import SentenceScorer
from langcheck.settings.loading import ServerSettings, load_settings
from langcheck.tagging.loaded import Trained
from tests.conftest import SyncTableScorer, WordTableTagger


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


def typo_llm(request: httpx.Request) -> httpx.Response:
    """Like ``fake_llm`` but finds "teh" in S1."""
    system = json.loads(request.content)["messages"][0]["content"]
    if "numbered sentences" not in system:
        return fake_llm(request)
    issue = {
        "sentence": "S1",
        "original": "teh",
        "kind": "Spelling",
        "explanation": "Misspelt.",
        "options": ["tea", "the"],
    }
    content = json.dumps({"issues": [issue]})
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_the_configured_scorer_ranks_options(tmp_path: Path) -> None:
    settings = load_settings(
        ServerSettings,
        overrides={
            "auth": {"mode": "none"},
            "database": {"url": f"sqlite:///{tmp_path / 'langcheck.db'}"},
            "llm": {"base_url": "http://llm.test/v1", "retries": 0},
            "scorer": {"adapters": ["pll"]},
        },
        yaml_data={},
    )
    scorer = SyncTableScorer({"Read the book.": 2.0, "Read tea book.": -1.0})

    def load(adapter: ScorerAdapter, config: ScorerConfig) -> SentenceScorer:
        return scorer

    app = build_app(
        lambda: open_services(settings, transport=httpx.MockTransport(typo_llm), load=load)
    )

    with TestClient(app) as client:
        response = client.post("/check", json={"text": "Read teh book."})

    [suggestion] = response.json()["suggestions"]
    assert suggestion["options"] == [
        {"replacement": "the", "score": 2.0, "delta": 2.0, "confidence": "high"},
        {"replacement": "tea", "score": -1.0, "delta": -1.0, "confidence": "high"},
    ]
    assert len(scorer.batches) == 1


def correcting_llm(request: httpx.Request) -> httpx.Response:
    """An LLM that describes every document the same way and corrects the tagger's flag."""
    system = json.loads(request.content)["messages"][0]["content"]
    if "error detector" in system:
        content = json.dumps(
            {
                "flags": [
                    {
                        "flag": "F1",
                        "corrections": [
                            {"replacement": "goes", "kind": "VerbForm", "explanation": "Agrees."},
                            {"replacement": "gone", "kind": "VerbForm", "explanation": "Past."},
                        ],
                    }
                ]
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    return fake_llm(request)


def test_a_configured_tagger_serves_one_shot_checks_and_documents(tmp_path: Path) -> None:
    settings = load_settings(
        ServerSettings,
        overrides={
            "auth": {"mode": "none"},
            "database": {"url": f"sqlite:///{tmp_path / 'langcheck.db'}"},
            "llm": {"base_url": "http://llm.test/v1", "retries": 0},
            "checker": {"debounce": 0},
            "correction": {"checkpoint": str(tmp_path / "tagger")},
        },
        yaml_data={},
    )

    def load(path: Path, *, device: str) -> Trained:
        assert path == tmp_path / "tagger"
        return Trained(WordTableTagger({"go": 0.9, "gone": 0.5}), 0.5, "base", 0.5)

    app = build_app(
        lambda: open_services(
            settings, transport=httpx.MockTransport(correcting_llm), load_tagger=load
        )
    )

    with TestClient(app) as client:
        response = client.post("/check", json={"text": "He go home."})
        [suggestion] = response.json()["suggestions"]
        client.put("/documents/a.md", json={"text": "He go home."})
        for _ in range(200):
            body = client.get("/documents/a.md/suggestions").json()
            if body["state"] != "checking" and body["suggestions"]:
                break
            time.sleep(0.01)

    assert suggestion["original"] == "go"
    assert suggestion["confidence"] == "high"
    assert [(o["replacement"], o["confidence"]) for o in suggestion["options"]] == [
        ("goes", "high"),
        ("gone", "low"),
    ]
    assert body["suggestions"][0]["id"] == suggestion["id"]


def test_a_configured_tagger_asks_the_llm_for_no_context_unless_told_to(tmp_path: Path) -> None:
    settings = load_settings(
        ServerSettings,
        overrides={
            "auth": {"mode": "none"},
            "database": {"url": f"sqlite:///{tmp_path / 'langcheck.db'}"},
            "llm": {"base_url": "http://llm.test/v1", "retries": 0},
            "correction": {"checkpoint": str(tmp_path / "tagger")},
        },
        yaml_data={},
    )
    calls: list[str] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["messages"][0]["content"][:30])
        return fake_llm(request)

    def load(path: Path, *, device: str) -> Trained:
        return Trained(WordTableTagger({}), 0.5, "base", 0.5)

    app = build_app(
        lambda: open_services(settings, transport=httpx.MockTransport(counting), load_tagger=load)
    )

    with TestClient(app) as client:
        checked = client.post("/check", json={"text": "All fine here."})
        assert checked.status_code == 200
        assert calls == []
        built = client.post("/context", json={"text": "All fine here."})

    assert built.json()["summary"] == "A note."
    assert len(calls) == 1
