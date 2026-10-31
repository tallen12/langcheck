"""The pygls adapter: wires LSP messages to :class:`LspProxy` and converts its values.

Text sync is incremental; pygls applies each change to its workspace and the proxy reads
the full text from there, so the service always gets whole versions.
"""

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from importlib.metadata import version

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from langcheck.check.suggestion import Confidence
from langcheck.lsp.models import (
    SOURCE,
    CodeAction,
    CommandAction,
    Diagnostic,
    ProxyCommand,
    ReplaceAction,
)
from langcheck.lsp.positions import Position, Range
from langcheck.lsp.proxy import LspProxy

type ProxyFactory = Callable[[LanguageServer], AbstractAsyncContextManager[LspProxy]]


class LspAdapter:
    """Handles the LSP requests the server registers, once a proxy is running.

    ``text`` reads a document's current full text (from pygls's workspace) and ``log``
    writes to the editor's output.
    """

    def __init__(
        self, proxy: LspProxy, *, text: Callable[[str], str], log: Callable[[str], None]
    ) -> None:
        self._proxy = proxy
        self._text = text
        self._log = log

    async def did_open(self, params: types.DidOpenTextDocumentParams) -> None:
        """Forward a newly opened document."""
        uri = params.text_document.uri
        await self._proxy.opened(uri, self._text(uri))

    async def did_change(self, params: types.DidChangeTextDocumentParams) -> None:
        """Forward the document's new full text."""
        uri = params.text_document.uri
        await self._proxy.changed(uri, self._text(uri))

    async def did_close(self, params: types.DidCloseTextDocumentParams) -> None:
        """Stop checking a closed document."""
        await self._proxy.closed(params.text_document.uri)

    def code_actions(self, params: types.CodeActionParams) -> list[types.CodeAction]:
        """Quick fixes for langcheck diagnostics in the requested range."""
        selection = from_lsp_range(params.range)
        return [
            to_lsp_action(a) for a in self._proxy.code_actions(params.text_document.uri, selection)
        ]

    async def command(self, name: ProxyCommand, arguments: Sequence[object]) -> None:
        """Run a proxy command; arguments are the strings the code action carried."""
        texts = [a for a in arguments if isinstance(a, str)]
        if len(texts) != 2:
            self._log(f"langcheck: {name.value} needs a URI and one value")
            return
        uri, value = texts
        match name:
            case ProxyCommand.DISMISS:
                await self._proxy.dismiss(uri, value)
            case ProxyCommand.ADD_TO_DICTIONARY:
                await self._proxy.add_to_dictionary(uri, value)

    async def shutdown(self) -> None:
        """Close every document on the service."""
        await self._proxy.shutdown()


def build_server(open_proxy: ProxyFactory) -> LanguageServer:
    """The language server. ``open_proxy`` opens the proxy (and its HTTP client) inside
    the server's event loop on ``initialized`` and closes it on ``shutdown``.
    """
    server = LanguageServer(
        "langcheck",
        version("langcheck"),
        text_document_sync_kind=types.TextDocumentSyncKind.Incremental,
    )
    running: list[LspAdapter] = []
    resources = AsyncExitStack()

    @server.feature(types.INITIALIZED)
    async def initialized(ls: LanguageServer, params: types.InitializedParams) -> None:
        proxy = await resources.enter_async_context(open_proxy(ls))
        running.append(
            LspAdapter(
                proxy,
                text=lambda uri: ls.workspace.get_text_document(uri).source,
                log=lambda line: log_to(ls, line),
            )
        )

    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    async def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
        for adapter in running:
            await adapter.did_open(params)

    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    async def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
        for adapter in running:
            await adapter.did_change(params)

    @server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
    async def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
        for adapter in running:
            await adapter.did_close(params)

    @server.feature(
        types.TEXT_DOCUMENT_CODE_ACTION,
        types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
    )
    def code_action(ls: LanguageServer, params: types.CodeActionParams) -> list[types.CodeAction]:
        return [action for adapter in running for action in adapter.code_actions(params)]

    @server.command(ProxyCommand.DISMISS.value)
    async def dismiss(ls: LanguageServer, *arguments: object) -> None:
        for adapter in running:
            await adapter.command(ProxyCommand.DISMISS, arguments)

    @server.command(ProxyCommand.ADD_TO_DICTIONARY.value)
    async def add_to_dictionary(ls: LanguageServer, *arguments: object) -> None:
        for adapter in running:
            await adapter.command(ProxyCommand.ADD_TO_DICTIONARY, arguments)

    @server.feature(types.SHUTDOWN)
    async def shutdown(ls: LanguageServer, params: None) -> None:
        for adapter in running:
            await adapter.shutdown()
        running.clear()
        await resources.aclose()

    return server


async def publish_to(server: LanguageServer, uri: str, diagnostics: list[Diagnostic]) -> None:
    """Send ``diagnostics`` for ``uri`` to the editor."""
    server.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=uri, diagnostics=[to_lsp_diagnostic(d) for d in diagnostics]
        )
    )


def log_to(server: LanguageServer, message: str) -> None:
    """Write ``message`` to the editor's output for this server."""
    server.window_log_message(types.LogMessageParams(types.MessageType.Warning, message))


def to_lsp_position(position: Position) -> types.Position:
    """The lsprotocol position."""
    return types.Position(line=position.line, character=position.character)


def to_lsp_range(range_: Range) -> types.Range:
    """The lsprotocol range."""
    return types.Range(start=to_lsp_position(range_.start), end=to_lsp_position(range_.end))


def from_lsp_range(range_: types.Range) -> Range:
    """The proxy's range."""
    return Range(
        Position(range_.start.line, range_.start.character),
        Position(range_.end.line, range_.end.character),
    )


def to_lsp_diagnostic(diagnostic: Diagnostic) -> types.Diagnostic:
    """A warning from ``langcheck``, or information when its confidence is low, whose code
    is the error kind.
    """
    severity = (
        types.DiagnosticSeverity.Warning
        if diagnostic.confidence is Confidence.HIGH
        else types.DiagnosticSeverity.Information
    )
    return types.Diagnostic(
        range=to_lsp_range(diagnostic.range),
        message=diagnostic.message,
        severity=severity,
        code=diagnostic.kind,
        source=SOURCE,
        data={"suggestion_id": diagnostic.suggestion_id},
    )


def to_lsp_action(action: CodeAction) -> types.CodeAction:
    """A quick fix with a workspace edit, or one that runs a proxy command."""
    match action:
        case ReplaceAction():
            edit = types.TextEdit(
                range=to_lsp_range(action.diagnostic.range), new_text=action.replacement
            )
            return types.CodeAction(
                title=action.title,
                kind=types.CodeActionKind.QuickFix,
                diagnostics=[to_lsp_diagnostic(action.diagnostic)],
                is_preferred=action.preferred,
                edit=types.WorkspaceEdit(changes={action.uri: [edit]}),
            )
        case CommandAction():
            return types.CodeAction(
                title=action.title,
                kind=types.CodeActionKind.QuickFix,
                diagnostics=[to_lsp_diagnostic(action.diagnostic)],
                command=types.Command(
                    title=action.title,
                    command=action.command.value,
                    arguments=list(action.arguments),
                ),
            )
