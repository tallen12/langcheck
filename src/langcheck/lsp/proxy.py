"""The LSP proxy's logic: keep the service in step with the editor's open documents.

The editor sends full text on every change. Only one update per document is in flight;
changes made meanwhile collapse into the latest text. The service answers each update
with its version, and its event stream sends snapshots per version; a snapshot becomes
diagnostics only when it belongs to the version whose text the proxy sent last, so
offsets always refer to the text the editor has.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langcheck.api.client import ServiceError
from langcheck.api.schemas import DocumentBody
from langcheck.lsp.models import (
    CodeAction,
    CommandAction,
    Diagnostic,
    ProxyCommand,
    ReplaceAction,
)
from langcheck.lsp.positions import LineIndex, Range
from langcheck.lsp.protocols import DocumentsService

DEFAULT_RETRY_DELAY = 2.0
KEPT_VERSIONS = 8

type Publish = Callable[[str, list[Diagnostic]], Awaitable[None]]
type Log = Callable[[str], None]


@dataclass(slots=True)
class _Open:
    """What the proxy knows about one open document."""

    text: str
    pending: str | None = None
    sending: asyncio.Task[None] | None = None
    version: int | None = None
    texts: dict[int, str] = field(default_factory=dict)
    snapshot: DocumentBody | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # The text ``diagnostics`` were placed on; their ranges hold only while the editor
    # still has exactly this text.
    shown: str | None = None
    watcher: asyncio.Task[None] | None = None


class LspProxy:
    """Forwards open documents to the service and its suggestions back as diagnostics."""

    def __init__(
        self,
        service: DocumentsService,
        *,
        publish: Publish,
        log: Log,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._service = service
        self._publish = publish
        self._log = log
        self._sleep = sleep
        self._retry_delay = retry_delay
        self._open: dict[str, _Open] = {}

    async def opened(self, uri: str, text: str) -> None:
        """The editor opened ``uri``."""
        self._open[uri] = _Open(text=text)
        self._queue(uri, text)

    async def changed(self, uri: str, text: str) -> None:
        """The editor's text of ``uri`` is now ``text``."""
        document = self._open.get(uri)
        if document is None:
            await self.opened(uri, text)
            return
        document.text = text
        self._queue(uri, text)

    async def closed(self, uri: str) -> None:
        """The editor closed ``uri``: stop following it and close it on the service."""
        document = self._open.pop(uri, None)
        if document is None:
            return
        for task in (document.watcher, document.sending):
            if task is not None:
                task.cancel()
        try:
            await self._service.close_document(uri)
        except ServiceError as error:
            self._log(f"langcheck: could not close {uri}: {error}")
        await self._publish(uri, [])

    def code_actions(self, uri: str, selection: Range) -> list[CodeAction]:
        """Quick fixes for the diagnostics that overlap ``selection``, options in ranked
        order, then "Add to dictionary" and "Dismiss". None while the editor's text has
        moved on from the text the diagnostics were placed on: their edits would replace
        the wrong characters.
        """
        document = self._open.get(uri)
        if document is None or document.shown != document.text:
            return []
        actions: list[CodeAction] = []
        for diagnostic in document.diagnostics:
            if not diagnostic.range.overlaps(selection):
                continue
            actions += [
                ReplaceAction(
                    title=f"Replace with '{option}'" if option else "Delete",
                    uri=uri,
                    diagnostic=diagnostic,
                    replacement=option,
                    preferred=index == 0,
                )
                for index, option in enumerate(diagnostic.options)
            ]
            if diagnostic.original.strip() and " " not in diagnostic.original.strip():
                actions.append(
                    CommandAction(
                        title=f"Add '{diagnostic.original.strip()}' to the dictionary",
                        diagnostic=diagnostic,
                        command=ProxyCommand.ADD_TO_DICTIONARY,
                        arguments=(uri, diagnostic.original.strip()),
                    )
                )
            actions.append(
                CommandAction(
                    title="Dismiss",
                    diagnostic=diagnostic,
                    command=ProxyCommand.DISMISS,
                    arguments=(uri, diagnostic.suggestion_id),
                )
            )
        return actions

    async def dismiss(self, uri: str, suggestion_id: str) -> None:
        """Dismiss a suggestion; the service's next snapshot drops it."""
        try:
            await self._service.dismiss(uri, suggestion_id)
        except ServiceError as error:
            self._log(f"langcheck: could not dismiss: {error}")

    async def add_to_dictionary(self, uri: str, word: str) -> None:
        """Add ``word`` to the document's dictionary; the service re-checks it."""
        try:
            await self._service.add_words(uri, [word])
        except ServiceError as error:
            self._log(f"langcheck: could not add {word!r} to the dictionary: {error}")

    async def shutdown(self) -> None:
        """Close every document the proxy still follows."""
        for uri in list(self._open):
            await self.closed(uri)

    def _queue(self, uri: str, text: str) -> None:
        document = self._open[uri]
        document.pending = text
        if document.sending is None or document.sending.done():
            document.sending = asyncio.create_task(self._send(uri, document))

    async def _send(self, uri: str, document: _Open) -> None:
        while document.pending is not None:
            text, document.pending = document.pending, None
            try:
                version = await self._service.put_document(uri, text)
            except ServiceError as error:
                self._log(f"langcheck: could not send {uri}: {error}")
                return
            document.version = version
            document.texts[version] = text
            for old in sorted(document.texts)[:-KEPT_VERSIONS]:
                del document.texts[old]
            if document.watcher is None or document.watcher.done():
                document.watcher = asyncio.create_task(self._follow(uri, document))
            await self._show(uri, document)

    async def _follow(self, uri: str, document: _Open) -> None:
        try:
            async for snapshot in self._service.events(uri):
                if snapshot is None:
                    break
                document.snapshot = snapshot
                await self._show(uri, document)
        except ServiceError as error:
            self._log(f"langcheck: lost events for {uri}: {error}")
        if self._open.get(uri) is not document:
            return
        # The stream ended while the editor still has the document open: the service
        # may have restarted and forgotten it, so send the text again, which also starts
        # a new event stream.
        await self._sleep(self._retry_delay)
        if self._open.get(uri) is document:
            self._queue(uri, document.text)

    async def _show(self, uri: str, document: _Open) -> None:
        snapshot = document.snapshot
        if snapshot is None or snapshot.version != document.version:
            return
        text = document.texts.get(snapshot.version)
        if text is None:
            return
        document.diagnostics = diagnostics_for(text, snapshot)
        document.shown = text
        await self._publish(uri, document.diagnostics)


def diagnostics_for(text: str, snapshot: DocumentBody) -> list[Diagnostic]:
    """The snapshot's suggestions as diagnostics on ``text``."""
    index = LineIndex(text)
    return [
        Diagnostic(
            range=index.range(s.start, s.end),
            message=_message(s.original, [o.replacement for o in s.options], s.explanation),
            kind=s.kind.value,
            suggestion_id=s.id,
            original=s.original,
            options=tuple(o.replacement for o in s.options),
        )
        for s in snapshot.suggestions
    ]


def _message(original: str, options: list[str], explanation: str) -> str:
    fixes = " or ".join(f"'{o}'" for o in options)
    head = f"'{original}' → {fixes}" if fixes else f"'{original}'"
    return f"{head}: {explanation}" if explanation else head
