import json
import time
from collections.abc import Collection
from urllib.parse import quote

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from langcheck.api.documents import document_router, sse
from langcheck.api.schemas import DocumentBody
from langcheck.auth.models import User
from langcheck.check.checker import CheckResult
from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.context.models import DocumentContext
from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.documents.events import Closed, Snapshot
from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.models import CheckState
from langcheck.documents.sessions import DocumentSessions
from langcheck.limits.user_limiter import Unlimited
from langcheck.llm.structured import Failed
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter

URI = "file:///home/ana/notes.md"
PATH = f"/documents/{quote(URI, safe='')}"


class TypoChecker:
    """Suggests "the" or "tea" for every "teh"."""

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        found = []
        # Like the real checker, issues on dictionary words are dropped.
        start = -1 if "teh" in dictionary else text.find("teh")
        while start != -1:
            span = Span(start, start + 3)
            found.append(
                Suggestion(
                    id=suggestion_id(span, "teh", ErrorKind.SPELLING),
                    span=span,
                    original="teh",
                    kind=ErrorKind.SPELLING,
                    operation=Operation.REPLACEMENT,
                    explanation="Misspelt.",
                    options=(Option("the"), Option("tea")),
                )
            )
            start = text.find("teh", start + 1)
        return CheckResult(tuple(found), (), ())


class NoContexts:
    async def build(self, text: str) -> DocumentContext | Failed:
        return DocumentContext.empty()


def make_client(
    *, keepalive: float = 15.0, dictionary: MemoryDictionaryStore | None = None
) -> TestClient:
    async def no_wait(seconds: float) -> None:
        return None

    words = dictionary if dictionary is not None else MemoryDictionaryStore()
    sessions = DocumentSessions(
        MemoryDocumentStore(),
        checker=TypoChecker(),
        contexts=NoContexts(),
        splitter=RegexSentenceSplitter(),
        limiter=Unlimited(),
        dictionary=words,
        sleep=no_wait,
    )

    async def caller(request: Request) -> User:
        return User(request.headers.get("x-user", "ana"))

    app = FastAPI()
    app.include_router(
        document_router(lambda: sessions, lambda: words, caller, keepalive=keepalive)
    )
    return TestClient(app)


def wait_done(client: TestClient, *, user: str = "ana") -> DocumentBody:
    for _ in range(200):
        response = client.get(f"{PATH}/suggestions", headers={"x-user": user})
        body = DocumentBody.model_validate_json(response.content)
        if body.state is CheckState.DONE:
            return body
        time.sleep(0.005)
    pytest.fail("the check never finished")


def test_open_check_and_read_suggestions() -> None:
    with make_client() as client:
        opened = client.put(PATH, json={"text": "Read teh book."})
        body = wait_done(client)

    assert opened.json() == {"version": 1}
    assert body.version == 1
    [suggestion] = body.suggestions
    assert (suggestion.start, suggestion.end) == (5, 8)


def test_versions_guard_updates_and_reads() -> None:
    with make_client() as client:
        client.put(PATH, json={"text": "One."})
        updated = client.put(PATH, json={"text": "Two.", "base_version": 1})
        stale = client.put(PATH, json={"text": "Three.", "base_version": 1})
        old_read = client.get(f"{PATH}/suggestions", params={"version": 1})

    assert updated.json() == {"version": 2}
    assert stale.status_code == 409
    assert stale.json() == {"detail": {"message": "version conflict", "current": 2}}
    assert old_read.status_code == 409


def test_accept_returns_the_edit_and_dismiss_hides_the_suggestion() -> None:
    with make_client() as client:
        client.put(PATH, json={"text": "Read teh book. Then teh end."})
        first, second = wait_done(client).suggestions
        accepted = client.post(
            f"{PATH}/suggestions/{first.id}/accept", json={"option": 1, "base_version": 1}
        )
        missing = client.post(f"{PATH}/suggestions/sg_nope/accept", json={})
        remaining = wait_done(client).suggestions[0].id
        dismissed = client.post(f"{PATH}/suggestions/{remaining}/dismiss")
        again = client.post(f"{PATH}/suggestions/{remaining}/dismiss")
        final = client.get(f"{PATH}/suggestions").json()

    assert accepted.json() == {"version": 2, "start": 5, "end": 8, "replacement": "tea"}
    assert second.start == 20
    assert missing.status_code == 404
    assert (dismissed.status_code, again.status_code) == (204, 404)
    assert final["suggestions"] == []


def test_close_rename_and_user_scoping() -> None:
    renamed_path = f"/documents/{quote('file:///home/ana/renamed.md', safe='')}"
    with make_client() as client:
        client.put(PATH, json={"text": "Hi."})
        other_user = client.get(f"{PATH}/suggestions", headers={"x-user": "ben"})
        renamed = client.post(f"{PATH}/rename", json={"uri": "file:///home/ana/renamed.md"})
        gone = client.get(f"{PATH}/suggestions")
        client.put(PATH, json={"text": "New."})
        clash = client.post(f"{PATH}/rename", json={"uri": "file:///home/ana/renamed.md"})
        missing = client.post("/documents/nope/rename", json={"uri": "x"})
        closed = client.delete(renamed_path)
        closed_again = client.delete(renamed_path)

    assert other_user.status_code == 404
    assert renamed.json() == {"version": 1}
    assert gone.status_code == 404
    assert clash.status_code == 409
    assert missing.status_code == 404
    assert (closed.status_code, closed_again.status_code) == (204, 404)


def test_events_stream_snapshots_of_the_open_document() -> None:
    with make_client() as client:
        assert client.get(f"{PATH}/events").status_code == 404
        client.put(PATH, json={"text": "Read teh book."})
        wait_done(client)
        with client.stream("GET", f"{PATH}/events") as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            lines = stream.iter_lines()
            assert next(lines) == "event: snapshot"
            data = json.loads(next(lines).removeprefix("data: "))

    assert data["version"] == 1
    assert data["state"] == "done"
    assert len(data["suggestions"]) == 1


def test_sse_formats_snapshots_and_closed() -> None:
    snapshot = sse(Snapshot(3, CheckState.CHECKING, (), ()))

    assert snapshot.startswith("event: snapshot\ndata: {")
    assert snapshot.endswith("\n\n")
    assert json.loads(snapshot.split("data: ", 1)[1])["state"] == "checking"
    assert sse(Closed()) == "event: closed\ndata: {}\n\n"


def test_dictionary_words_hide_suggestions_and_come_back_when_removed() -> None:
    with make_client() as client:
        client.put(PATH, json={"text": "Read teh book."})
        assert len(wait_done(client).suggestions) == 1
        added = client.post(f"{PATH}/dictionary", json={"words": ["teh", "Bilbo"]})
        hidden = wait_done(client)
        listed = client.get(f"{PATH}/dictionary")
        removed = client.request("DELETE", f"{PATH}/dictionary", json={"words": ["teh"]})
        back = wait_done(client)
        still_open = client.get(f"{PATH}/suggestions")

    assert added.json() == {"words": ["Bilbo", "teh"]}
    assert hidden.suggestions == []
    assert listed.json() == {"words": ["Bilbo", "teh"]}
    assert removed.json() == {"words": ["Bilbo"]}
    assert len(back.suggestions) == 1
    assert still_open.status_code == 200


def test_dictionaries_work_for_documents_that_are_not_open() -> None:
    with make_client() as client:
        added = client.post(f"{PATH}/dictionary", json={"words": ["Shire"]})
        blank = client.post(f"{PATH}/dictionary", json={"words": ["  "]})
        empty = client.post(f"{PATH}/dictionary", json={"words": []})

    assert added.json() == {"words": ["Shire"]}
    assert (blank.status_code, empty.status_code) == (422, 422)


def test_renaming_a_document_moves_its_dictionary() -> None:
    dictionary = MemoryDictionaryStore()
    with make_client(dictionary=dictionary) as client:
        client.put(PATH, json={"text": "Hi."})
        client.post(f"{PATH}/dictionary", json={"words": ["Shire"]})
        client.post(f"{PATH}/rename", json={"uri": "file:///home/ana/new.md"})

    assert dictionary.words("ana", URI) == frozenset()
    assert dictionary.words("ana", "file:///home/ana/new.md") == {"Shire"}
