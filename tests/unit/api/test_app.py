from collections.abc import AsyncGenerator, Collection, Mapping
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from langcheck.api.app import Services, build_app
from langcheck.auth.models import AuthResult, Unauthenticated, User
from langcheck.check.checker import CheckResult, Unchecked
from langcheck.check.suggestion import Option, Suggestion
from langcheck.context.models import DocumentContext, Register
from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.sessions import DocumentSessions
from langcheck.limits.protocols import CheckLimiter
from langcheck.limits.user_limiter import Unlimited, UserLimiter
from langcheck.llm.structured import Failed
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter

CONTEXT = DocumentContext(
    summary="Notes.",
    register=Register.INFORMAL,
    dialect="en-GB",
    notes="",
)
SUGGESTION = Suggestion(
    id="sg_1",
    span=Span(5, 8),
    original="teh",
    kind=ErrorKind.SPELLING,
    operation=Operation.REPLACEMENT,
    explanation="Misspelt.",
    options=(Option("the"),),
)


class KeyAuthenticator:
    """Lets in ``Bearer good`` as ana."""

    async def authenticate(self, headers: Mapping[str, str]) -> AuthResult:
        if headers.get("authorization") == "Bearer good":
            return User("ana")
        return Unauthenticated("invalid API key")


class RecordingChecker:
    """Returns one suggestion and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, DocumentContext, list[str]]] = []

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        self.calls.append((text, user, context, list(dictionary)))
        return CheckResult((SUGGESTION,), (Unchecked(Span(10, 12), "LLM error: down"),), ())


class FixedContexts:
    """Builds ``CONTEXT``, or fails when told to."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.built: list[str] = []

    async def build(self, text: str) -> DocumentContext | Failed:
        self.built.append(text)
        return Failed("no reply") if self.fail else CONTEXT


def client_for(
    checker: RecordingChecker, contexts: FixedContexts, *, limiter: CheckLimiter | None = None
) -> TestClient:
    dictionary = MemoryDictionaryStore()
    dictionary.add("ana", "file:///notes.md", ["Shire", "Frodo"])

    @asynccontextmanager
    async def services() -> AsyncGenerator[Services]:
        yield Services(
            authenticator=KeyAuthenticator(),
            checker=checker,
            contexts=contexts,
            limiter=limiter or Unlimited(),
            documents=DocumentSessions(
                MemoryDocumentStore(),
                checker=checker,
                contexts=contexts,
                splitter=RegexSentenceSplitter(),
                limiter=Unlimited(),
                dictionary=dictionary,
            ),
            dictionary=dictionary,
        )

    return TestClient(build_app(services))


AUTH = {"Authorization": "Bearer good"}


@pytest.fixture
def checker() -> RecordingChecker:
    return RecordingChecker()


def test_health_needs_no_authentication(checker: RecordingChecker) -> None:
    with client_for(checker, FixedContexts()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/check", "/context"])
def test_endpoints_refuse_unauthenticated_requests(checker: RecordingChecker, path: str) -> None:
    with client_for(checker, FixedContexts()) as client:
        response = client.post(path, json={"text": "x"}, headers={"Authorization": "Bearer bad"})

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid API key"}
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert checker.calls == []


def test_check_builds_context_and_returns_suggestions(checker: RecordingChecker) -> None:
    contexts = FixedContexts()
    with client_for(checker, contexts) as client:
        response = client.post(
            "/check", json={"text": "Read teh book.", "dictionary": ["Shire"]}, headers=AUTH
        )

    assert response.status_code == 200
    assert response.json() == {
        "suggestions": [
            {
                "id": "sg_1",
                "start": 5,
                "end": 8,
                "original": "teh",
                "kind": "Spelling",
                "operation": "R",
                "explanation": "Misspelt.",
                "options": [
                    {"replacement": "the", "score": None, "delta": None, "confidence": "high"}
                ],
                "confidence": "high",
            }
        ],
        "unchecked": [{"start": 10, "end": 12, "reason": "LLM error: down"}],
        "context": {
            "summary": "Notes.",
            "register": "informal",
            "dialect": "en-GB",
            "notes": "",
        },
    }
    assert checker.calls == [("Read teh book.", "ana", CONTEXT, ["Shire"])]
    assert contexts.built == ["Read teh book."]


def test_check_uses_a_given_context_without_building_one(checker: RecordingChecker) -> None:
    contexts = FixedContexts()
    given = {
        "summary": "S",
        "register": "formal",
        "dialect": "en-US",
        "notes": "",
    }
    with client_for(checker, contexts) as client:
        response = client.post("/check", json={"text": "Hi.", "context": given}, headers=AUTH)

    assert response.status_code == 200
    assert contexts.built == []
    assert checker.calls[0][2] == DocumentContext("S", Register.FORMAL, "en-US", "")


def test_check_goes_ahead_without_context_when_it_cannot_be_built(
    checker: RecordingChecker,
) -> None:
    with client_for(checker, FixedContexts(fail=True)) as client:
        response = client.post("/check", json={"text": "Hi."}, headers=AUTH)

    assert response.status_code == 200
    assert checker.calls[0][2] == DocumentContext.empty()


def test_check_rejects_malformed_bodies(checker: RecordingChecker) -> None:
    with client_for(checker, FixedContexts()) as client:
        missing = client.post("/check", json={}, headers=AUTH)
        extra = client.post("/check", json={"text": "x", "rewrite": True}, headers=AUTH)

    assert (missing.status_code, extra.status_code) == (422, 422)


def test_context_returns_the_built_context_or_502(checker: RecordingChecker) -> None:
    with client_for(checker, FixedContexts()) as client:
        built = client.post("/context", json={"text": "Hi."}, headers=AUTH)
    with client_for(checker, FixedContexts(fail=True)) as client:
        failed = client.post("/context", json={"text": "Hi."}, headers=AUTH)

    assert built.status_code == 200
    assert built.json()["register"] == "informal"
    assert failed.status_code == 502
    assert failed.json() == {"detail": "no reply"}


def test_check_answers_429_when_the_user_is_over_the_limit(checker: RecordingChecker) -> None:
    limiter = UserLimiter(max_in_flight=1, per_minute=1, clock=lambda: 100.0)
    with client_for(checker, FixedContexts(), limiter=limiter) as client:
        first = client.post("/check", json={"text": "Hi."}, headers=AUTH)
        second = client.post("/check", json={"text": "Hi."}, headers=AUTH)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"
    assert len(checker.calls) == 1


def test_check_adds_the_stored_dictionary_of_the_named_document(
    checker: RecordingChecker,
) -> None:
    contexts = FixedContexts()
    with client_for(checker, contexts) as client:
        client.post(
            "/check",
            json={"text": "Hi.", "dictionary": ["Bilbo"], "uri": "file:///notes.md"},
            headers=AUTH,
        )

    assert sorted(checker.calls[0][3]) == ["Bilbo", "Frodo", "Shire"]
