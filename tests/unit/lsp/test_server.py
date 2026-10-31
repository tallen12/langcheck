import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

from lsprotocol import types
from pygls.lsp.server import LanguageServer
from pygls.workspace import Workspace

from langcheck.api.schemas import DocumentBody
from langcheck.check.suggestion import Confidence
from langcheck.lsp.models import CommandAction, Diagnostic, ProxyCommand, ReplaceAction
from langcheck.lsp.positions import Position, Range
from langcheck.lsp.proxy import LspProxy
from langcheck.lsp.server import (
    LspAdapter,
    build_server,
    from_lsp_range,
    to_lsp_action,
    to_lsp_diagnostic,
    to_lsp_range,
)

URI = "file:///a.md"
DIAGNOSTIC = Diagnostic(
    Range(Position(0, 5), Position(0, 8)), "'teh' → 'the'", "Spelling", "sg_1", "teh", ("the",)
)


class RecordingService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def put_document(self, uri: str, text: str) -> int:
        self.calls.append(("put", uri, text))
        return 1

    async def close_document(self, uri: str) -> bool:
        self.calls.append(("close", uri))
        return True

    async def dismiss(self, uri: str, suggestion_id: str) -> bool:
        self.calls.append(("dismiss", uri, suggestion_id))
        return True

    async def add_words(self, uri: str, words: Sequence[str]) -> list[str]:
        self.calls.append(("add", uri, *words))
        return list(words)

    async def events(self, uri: str) -> AsyncIterator[DocumentBody | None]:
        yield None


def test_conversions_round_trip_ranges_and_mark_the_source() -> None:
    lsp_range = to_lsp_range(DIAGNOSTIC.range)

    assert from_lsp_range(lsp_range) == DIAGNOSTIC.range
    diagnostic = to_lsp_diagnostic(DIAGNOSTIC)
    assert (diagnostic.source, diagnostic.code) == ("langcheck", "Spelling")
    assert diagnostic.severity is types.DiagnosticSeverity.Warning
    assert diagnostic.data == {"suggestion_id": "sg_1"}


def test_actions_become_workspace_edits_or_commands() -> None:
    replace = to_lsp_action(
        ReplaceAction("Replace with 'the'", URI, DIAGNOSTIC, "the", preferred=True)
    )
    command = to_lsp_action(
        CommandAction("Dismiss", DIAGNOSTIC, ProxyCommand.DISMISS, (URI, "sg_1"))
    )

    assert replace.is_preferred is True
    assert replace.edit is not None
    assert replace.edit.changes is not None
    [edit] = replace.edit.changes[URI]
    assert (edit.new_text, edit.range) == ("the", to_lsp_range(DIAGNOSTIC.range))
    assert command.command is not None
    assert (command.command.command, command.command.arguments) == (
        "langcheck.dismiss",
        [URI, "sg_1"],
    )


def test_adapter_forwards_documents_from_the_workspace_and_runs_commands() -> None:
    workspace = Workspace(None)
    workspace.put_text_document(
        types.TextDocumentItem(uri=URI, language_id="markdown", version=1, text="Read teh book.")
    )
    service = RecordingService()
    logs: list[str] = []
    identifier = types.VersionedTextDocumentIdentifier(uri=URI, version=2)

    async def publish(uri: str, diagnostics: list[Diagnostic]) -> None:
        return None

    async def run() -> list[types.CodeAction]:
        proxy = LspProxy(service, publish=publish, log=logs.append)
        adapter = LspAdapter(
            proxy, text=lambda uri: workspace.get_text_document(uri).source, log=logs.append
        )
        await adapter.did_open(
            types.DidOpenTextDocumentParams(
                text_document=types.TextDocumentItem(
                    uri=URI, language_id="markdown", version=1, text="ignored"
                )
            )
        )
        await asyncio.sleep(0)
        await adapter.did_change(
            types.DidChangeTextDocumentParams(text_document=identifier, content_changes=[])
        )
        await asyncio.sleep(0)
        actions = adapter.code_actions(
            types.CodeActionParams(
                text_document=types.TextDocumentIdentifier(uri=URI),
                range=to_lsp_range(DIAGNOSTIC.range),
                context=types.CodeActionContext(diagnostics=[]),
            )
        )
        await adapter.command(ProxyCommand.DISMISS, [URI, "sg_1"])
        await adapter.command(ProxyCommand.ADD_TO_DICTIONARY, [URI, "teh"])
        await adapter.command(ProxyCommand.DISMISS, [URI])
        await adapter.did_close(
            types.DidCloseTextDocumentParams(text_document=types.TextDocumentIdentifier(uri=URI))
        )
        await adapter.shutdown()
        return actions

    actions = asyncio.run(run())

    assert actions == []
    assert service.calls[0] == ("put", URI, "Read teh book.")
    assert ("dismiss", URI, "sg_1") in service.calls
    assert ("add", URI, "teh") in service.calls
    assert service.calls[-1] == ("close", URI)
    assert logs == ["langcheck: langcheck.dismiss needs a URI and one value"]


def test_build_server_registers_the_features_and_commands() -> None:
    @asynccontextmanager
    async def open_proxy(ls: LanguageServer) -> AsyncGenerator[LspProxy]:
        raise AssertionError("not opened until initialized")
        yield  # pragma: no cover

    server = build_server(open_proxy)

    features = server.protocol.fm.features
    for method in [
        types.INITIALIZED,
        types.TEXT_DOCUMENT_DID_OPEN,
        types.TEXT_DOCUMENT_DID_CHANGE,
        types.TEXT_DOCUMENT_DID_CLOSE,
        types.TEXT_DOCUMENT_CODE_ACTION,
        types.SHUTDOWN,
    ]:
        assert method in features
    assert set(server.protocol.fm.commands) >= {"langcheck.dismiss", "langcheck.addToDictionary"}


def test_a_low_confidence_diagnostic_is_only_information() -> None:
    low = replace(DIAGNOSTIC, confidence=Confidence.LOW)

    assert to_lsp_diagnostic(low).severity is types.DiagnosticSeverity.Information
