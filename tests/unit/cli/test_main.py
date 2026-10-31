import asyncio
import json
import re
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pygls.lsp.server import LanguageServer

from langcheck.cli.environment import CliEnvironment
from langcheck.cli.main import CLEAN, FAILED, FOUND, build_app, nested_overrides
from tests.conftest import TokenDiffAnnotator

CONTEXT = {
    "summary": "A note.",
    "register": "neutral",
    "dialect": "en-US",
    "notes": "",
}
SERVICE_RESPONSE = {
    "suggestions": [
        {
            "id": "sg_1",
            "start": 5,
            "end": 8,
            "original": "teh",
            "kind": "Spelling",
            "operation": "R",
            "explanation": "Misspelt.",
            "options": [{"replacement": "the"}],
        }
    ],
    "unchecked": [],
    "context": CONTEXT,
}


def llm_reply(request: httpx.Request) -> httpx.Response:
    """An LLM that describes every document the same way and fixes "teh"."""
    messages = json.loads(request.content)["messages"]
    if "numbered sentences" in messages[0]["content"]:
        issues = [
            {
                "sentence": sentence_id,
                "original": "teh",
                "kind": "Spelling",
                "explanation": "Misspelt.",
                "options": ["the"],
            }
            for sentence_id, text in re.findall(r"^\[(S\d+)\] (.*)$", messages[1]["content"], re.M)
            if "teh" in text.split()
        ]
        content = json.dumps({"issues": issues})
    else:
        content = json.dumps(CONTEXT)
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@dataclass
class Network:
    """The LLM at llm.test and a langcheck service at svc.test."""

    requests: list[httpx.Request] = field(default_factory=list)
    service_status: int = 200
    words: set[str] = field(default_factory=set)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "llm.test":
            return llm_reply(request)
        if self.service_status != 200:
            return httpx.Response(self.service_status, json={"detail": "invalid API key"})
        if request.url.path.endswith("/dictionary"):
            if request.method == "POST":
                self.words |= set(json.loads(request.content)["words"])
            elif request.method == "DELETE":
                self.words -= set(json.loads(request.content)["words"])
            return httpx.Response(200, json={"words": sorted(self.words)})
        if request.url.path == "/check":
            return httpx.Response(200, json=SERVICE_RESPONSE)
        return httpx.Response(200, json=CONTEXT)


@dataclass
class RecordingRunner:
    """Records the server it was asked to run instead of running it."""

    runs: list[tuple[FastAPI, str, int]] = field(default_factory=list)

    def run(self, app: FastAPI, *, host: str, port: int) -> None:
        self.runs.append((app, host, port))


class ServedArchive:
    """A Downloader that writes a prepared archive."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.urls: list[str] = []

    def download(self, url: str, destination: Path) -> None:
        self.urls.append(url)
        destination.write_bytes(self.body)


@dataclass
class RecordingLanguageServers:
    """Records the language server it was asked to run."""

    servers: list[LanguageServer] = field(default_factory=list)

    def run(self, server: LanguageServer) -> None:
        self.servers.append(server)


class EditingWatcher:
    """A ``FileWatcher`` that writes each scripted text to the file, then reports it."""

    def __init__(self, edits: list[str]) -> None:
        self.edits = edits

    async def changes(self, path: Path) -> AsyncIterator[None]:
        for text in self.edits:
            await asyncio.to_thread(path.write_text, text, encoding="utf-8")
            yield None


@dataclass
class Cli:
    tmp_path: Path
    network: Network = field(default_factory=Network)
    runner: RecordingRunner = field(default_factory=RecordingRunner)
    downloader: ServedArchive = field(default_factory=lambda: ServedArchive(b""))
    language_servers: RecordingLanguageServers = field(
        default_factory=lambda: RecordingLanguageServers()
    )
    watcher: EditingWatcher = field(default_factory=lambda: EditingWatcher([]))
    out: list[str] = field(default_factory=list)
    err: list[str] = field(default_factory=list)

    def environment(self) -> CliEnvironment:
        return CliEnvironment(
            environ={},
            cwd=self.tmp_path,
            out=self.out.append,
            err=self.err.append,
            runner=self.runner,
            watcher=self.watcher,
            language_servers=self.language_servers,
            downloader=self.downloader,
            edit_annotators=TokenDiffAnnotator,
            clock=lambda: datetime(2026, 10, 28, tzinfo=UTC),
            transport=httpx.MockTransport(self.network),
        )

    def __call__(self, *argv: str) -> int:
        code = build_app(self.environment())(list(argv))
        assert isinstance(code, int)
        return code

    @property
    def output(self) -> str:
        return "\n".join(self.out)


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    for name in ["LANGCHECK_SERVER__URL", "LANGCHECK_SERVER__API_KEY", "LANGCHECK_LLM__MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHECK_LLM__BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LANGCHECK_LLM__RETRIES", "0")
    monkeypatch.setenv("LANGCHECK_DATABASE__URL", f"sqlite:///{tmp_path / 'langcheck.db'}")
    return Cli(tmp_path)


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    path = tmp_path / "notes.md"
    path.write_text("Read teh book.\n\nAll good here.\n", encoding="utf-8")
    return path


def test_check_without_a_server_runs_locally(cli: Cli, notes: Path) -> None:
    code = cli("check", str(notes))

    assert code == FOUND
    assert f"{notes}:1:6: Spelling: 'teh' -> 'the'" in cli.output
    assert "1 suggestion" in cli.output
    assert {r.url.host for r in cli.network.requests} == {"llm.test"}


def test_check_prints_json(cli: Cli, notes: Path) -> None:
    assert cli("check", str(notes), "--json", "--llm-model", "gemma3") == FOUND

    body = json.loads(cli.output)
    assert [s["original"] for s in body["suggestions"]] == ["teh"]
    assert body["context"]["register"] == "neutral"
    assert json.loads(cli.network.requests[0].content)["model"] == "gemma3"


def test_check_uses_the_server_with_its_api_key(
    cli: Cli, notes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHECK_SERVER__API_KEY", "lc_key")

    code = cli("check", str(notes), "--server", "http://svc.test")

    assert code == FOUND
    [request] = cli.network.requests
    assert str(request.url) == "http://svc.test/check"
    assert request.headers["Authorization"] == "Bearer lc_key"
    assert json.loads(request.content)["uri"] == notes.resolve().as_uri()
    assert "Spelling: 'teh' -> 'the'" in cli.output


def test_local_overrides_a_configured_server(cli: Cli, notes: Path) -> None:
    cli("check", str(notes), "--server", "http://svc.test", "--local")

    assert {r.url.host for r in cli.network.requests} == {"llm.test"}


def test_clean_text_exits_zero(cli: Cli, tmp_path: Path) -> None:
    clean = tmp_path / "clean.md"
    clean.write_text("All good.\n", encoding="utf-8")

    assert cli("check", str(clean)) == CLEAN
    assert cli.output.endswith("0 suggestions")


def test_failures_are_reported_with_exit_code_two(cli: Cli, notes: Path, tmp_path: Path) -> None:
    cli.network.service_status = 401

    assert cli("check", str(tmp_path / "missing.md")) == FAILED
    assert cli("check", str(notes), "--server", "http://svc.test") == FAILED
    assert cli("check", str(notes), "--config", str(tmp_path / "nope.yaml")) == FAILED
    assert len(cli.err) == 3
    assert "HTTP 401" in cli.err[1]
    assert "settings file not found" in cli.err[2]


def test_context_prints_the_document_context(cli: Cli, notes: Path) -> None:
    assert cli("context", str(notes)) == CLEAN
    assert "summary: A note." in cli.output

    cli.out.clear()
    assert cli("context", str(notes), "--json", "--server", "http://svc.test") == CLEAN
    assert json.loads(cli.output) == CONTEXT


def test_config_show_merges_sources_and_masks_secrets(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "langcheck.yaml"
    config.write_text("llm:\n  model: qwen3:8b\n  api_key: from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("LANGCHECK_LLM__MODEL", "llama3")

    assert cli("config", "show") == CLEAN

    assert "model: llama3" in cli.output
    assert "from-yaml" not in cli.output
    assert "api_key: '**********'" in cli.output
    assert "auth" not in cli.output


def test_config_show_server(cli: Cli) -> None:
    assert cli("config", "show", "--server") == CLEAN
    assert "mode: api_key" in cli.output


def test_serve_runs_the_app_with_the_settings(cli: Cli) -> None:
    assert cli("serve", "--port", "9000") == CLEAN

    [(app, host, port)] = cli.runner.runs
    assert isinstance(app, FastAPI)
    assert (host, port) == ("127.0.0.1", 9000)


def test_admin_adds_lists_and_revokes_users(cli: Cli) -> None:
    assert cli("admin", "add-user", "ana") == CLEAN
    token = cli.out.pop()
    assert token.startswith("lc_")

    assert cli("admin", "add-user", "ana") == FAILED
    assert cli("admin", "users") == CLEAN
    assert cli.out.pop() == "ana"
    assert cli("admin", "revoke", "ana") == CLEAN
    assert cli("admin", "revoke", "ana") == FAILED
    assert "ana has no API key" in cli.err[-1]


def test_nested_overrides_skip_unset_options() -> None:
    assert nested_overrides({"llm.model": "m", "llm.base_url": None, "server.url": "u"}) == {
        "llm": {"model": "m"},
        "server": {"url": "u"},
    }


def test_check_reports_a_configured_scorer_without_pytorch(
    cli: Cli, notes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGCHECK_SCORER__ADAPTERS", '["pll"]')
    monkeypatch.setitem(sys.modules, "langcheck.scoring.pll", None)

    assert cli("check", str(notes)) == FAILED
    assert "just setup cpu" in cli.err[-1]

    assert cli("check", str(notes), "--no-score") == FOUND


def test_watch_reports_new_and_resolved_suggestions(cli: Cli, notes: Path) -> None:
    cli.watcher.edits = [
        "Read the book.\n\nAll good here.\n",
        "Read the book.\n\nAll teh good here.\n",
    ]

    assert cli("watch", str(notes)) == CLEAN

    assert cli.out == [
        f"+ {notes}:1:6: Spelling: 'teh' -> 'the'",
        "1 suggestion",
        f"- {notes}: Spelling: 'teh' resolved",
        "0 suggestions",
        f"+ {notes}:3:5: Spelling: 'teh' -> 'the'",
        "1 suggestion",
    ]


def test_dict_lists_adds_and_removes_words_on_the_server(cli: Cli, notes: Path) -> None:
    server = ["--server", "http://svc.test"]

    assert cli("dict", str(notes), "add", "Bilbo", "Shire", *server) == CLEAN
    assert cli.out == ["Bilbo", "Shire"]
    cli.out.clear()
    assert cli("dict", str(notes), "remove", "Shire", *server) == CLEAN
    assert cli("dict", str(notes), *server) == CLEAN

    assert cli.out == ["Bilbo", "Bilbo"]
    uri = notes.resolve().as_uri()
    assert all(r.url.path == f"/documents/{uri}/dictionary" for r in cli.network.requests)
    assert [r.method for r in cli.network.requests] == ["POST", "DELETE", "GET"]


def test_dict_needs_a_server_and_words(cli: Cli, notes: Path) -> None:
    assert cli("dict", str(notes)) == FAILED
    assert "needs a server" in cli.err[-1]
    assert cli("dict", str(notes), "add", "--server", "http://svc.test") == FAILED
    assert "needs at least one word" in cli.err[-1]
    assert cli.network.requests == []


def test_lsp_runs_a_language_server_for_the_configured_server(cli: Cli) -> None:
    assert cli("lsp") == FAILED
    assert "lsp needs a server" in cli.err[-1]

    assert cli("lsp", "--server", "http://svc.test") == CLEAN
    [server] = cli.language_servers.servers
    assert server.name == "langcheck"


CONLL_M2 = """S Read teh book .
A 1 2|||R:SPELL|||the|||REQUIRED|||-NONE-|||0

S All good here .
"""


def conll_archive() -> bytes:
    import io
    import tarfile

    from langcheck.evaluation.datasets import SOURCES, Dataset

    data = CONLL_M2.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(SOURCES[Dataset.CONLL14_TEST].members[0])
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_evaluate_fetches_runs_shows_compares_and_tunes(cli: Cli, tmp_path: Path) -> None:
    assert cli("evaluate", "run", "conll14-test") == FAILED
    assert "evaluate fetch conll14-test" in cli.err[-1]

    cli.downloader.body = conll_archive()
    assert cli("evaluate", "fetch", "conll14-test") == CLEAN
    assert "for research use" in cli.out[0]

    cli.out.clear()
    assert cli("evaluate", "run", "conll14-test", "--output", str(tmp_path / "a.json")) == CLEAN
    assert "conll14-test: 2 sentences" in cli.output
    report = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert (report["overall"]["tp"], report["overall"]["fp"], report["overall"]["fn"]) == (1, 0, 0)
    assert report["scorer"] == "llm-order"

    cli.out.clear()
    assert cli("evaluate", "show", str(tmp_path / "a.json")) == CLEAN
    assert cli("evaluate", "compare", str(tmp_path / "a.json"), str(tmp_path / "a.json")) == CLEAN
    assert "+0.000" in cli.output

    assert cli("evaluate", "tune", str(tmp_path / "a.json")) == FAILED
    assert "no scored suggestions" in cli.err[-1]


def test_evaluate_tune_prints_the_policy(cli: Cli, tmp_path: Path) -> None:
    from langcheck.evaluation.report import KindScore, Latency, Outcome, Report
    from langcheck.evaluation.runner import Effect
    from langcheck.evaluation.scoring import Counts
    from langcheck.taxonomy.kinds import ErrorKind

    report = Report(
        dataset="conll14-test",
        model="m",
        prompt_version="check-v1",
        scorer="pll",
        sentences=5,
        overall=KindScore.of("overall", Counts(2, 3, 1)),
        kinds=[KindScore.of("Punctuation", Counts(2, 3, 1))],
        rejections={},
        unchecked=0,
        filtered=0,
        latency=Latency.of([1.0]),
        outcomes=[
            Outcome(
                kind=ErrorKind.PUNCTUATION, delta=d, effect=Effect.CORRECT if c else Effect.WRONG
            )
            for d, c in [(-1.0, False), (-0.5, False), (0.2, False), (1.0, True), (2.0, True)]
        ],
    )
    path = tmp_path / "scored.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")

    assert cli("evaluate", "tune", str(path)) == CLEAN
    assert "too few to tune" in cli.output
    cli.out.clear()

    assert cli("evaluate", "tune", str(path), "--min-suggestions", "5") == CLEAN

    assert "Punctuation            filter above 0.200" in cli.output
    assert cli.output.endswith("    Punctuation:\n      filter: true\n      margin: 0.2")
    cli.out.clear()

    assert cli("evaluate", "tune", str(path), "--objective", "closeness") == FAILED
    assert "no closeness; rerun it" in cli.err[-1]
