"""``langcheck lsp`` as an editor runs it: a subprocess on stdio, talking to a real HTTP
service on a local socket.
"""

import asyncio
import sys
import threading
from collections.abc import Collection, Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request
from lsprotocol import types
from pygls.lsp.client import LanguageClient

from langcheck.api.documents import document_router
from langcheck.auth.models import User
from langcheck.check.checker import CheckResult
from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.context.models import DocumentContext
from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.sessions import DocumentSessions
from langcheck.limits.user_limiter import Unlimited
from langcheck.llm.structured import Failed
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter

URI = "file:///tmp/notes.md"


class TypoChecker:
    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        start = text.find("teh")
        if start == -1 or "teh" in dictionary:
            return CheckResult((), (), ())
        span = Span(start, start + 3)
        found = Suggestion(
            id=suggestion_id(span, "teh", ErrorKind.SPELLING),
            span=span,
            original="teh",
            kind=ErrorKind.SPELLING,
            operation=Operation.REPLACEMENT,
            explanation="Misspelt.",
            options=(Option("the"),),
        )
        return CheckResult((found,), (), ())


class NoContexts:
    async def build(self, text: str) -> DocumentContext | Failed:
        return DocumentContext.empty()


@pytest.fixture
def service_url() -> Iterator[str]:
    """A langcheck documents service on a free local port, in a background thread."""

    async def no_wait(seconds: float) -> None:
        return None

    dictionary = MemoryDictionaryStore()
    sessions = DocumentSessions(
        MemoryDocumentStore(),
        checker=TypoChecker(),
        contexts=NoContexts(),
        splitter=RegexSentenceSplitter(),
        limiter=Unlimited(),
        dictionary=dictionary,
        sleep=no_wait,
    )

    async def caller(request: Request) -> User:
        return User("ana")

    app = FastAPI()
    app.include_router(document_router(lambda: sessions, lambda: dictionary, caller))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_an_editor_gets_diagnostics_and_quick_fixes(service_url: str) -> None:
    async def run() -> tuple[list[types.Diagnostic], list[types.CodeAction | types.Command]]:
        client = LanguageClient("test-editor", "1")
        published: asyncio.Queue[types.PublishDiagnosticsParams] = asyncio.Queue()

        def diagnostics(params: types.PublishDiagnosticsParams) -> None:
            published.put_nowait(params)

        # Registered by call: pygls's decorator is untyped.
        client.feature(types.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)(diagnostics)

        await client.start_io(
            sys.executable,
            "-c",
            "from langcheck.cli.main import main; main()",
            "lsp",
            "--server",
            service_url,
        )
        await client.initialize_async(
            types.InitializeParams(capabilities=types.ClientCapabilities())
        )
        client.initialized(types.InitializedParams())
        client.text_document_did_open(
            types.DidOpenTextDocumentParams(
                text_document=types.TextDocumentItem(
                    uri=URI, language_id="markdown", version=1, text="Read teh book."
                )
            )
        )
        found: list[types.Diagnostic] = []
        while not found:
            found = list((await asyncio.wait_for(published.get(), timeout=20)).diagnostics)
        actions = await client.text_document_code_action_async(
            types.CodeActionParams(
                text_document=types.TextDocumentIdentifier(uri=URI),
                range=found[0].range,
                context=types.CodeActionContext(diagnostics=found),
            )
        )
        await client.shutdown_async(None)
        client.exit(None)
        await client.stop()
        return found, list(actions or [])

    found, actions = asyncio.run(run())

    [diagnostic] = found
    assert diagnostic.range == types.Range(types.Position(0, 5), types.Position(0, 8))
    assert diagnostic.source == "langcheck"
    assert [a.title for a in actions] == [
        "Replace with 'the'",
        "Add 'teh' to the dictionary",
        "Dismiss",
    ]
