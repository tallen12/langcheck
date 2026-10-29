import asyncio
from collections.abc import AsyncIterator, Sequence

from langcheck.api.client import ServiceError
from langcheck.api.schemas import DocumentBody, OptionBody, SuggestionBody
from langcheck.documents.models import CheckState
from langcheck.lsp.models import CommandAction, Diagnostic, ProxyCommand, ReplaceAction
from langcheck.lsp.positions import Position, Range
from langcheck.lsp.protocols import DocumentsService
from langcheck.lsp.proxy import LspProxy, diagnostics_for
from langcheck.taxonomy.kinds import ErrorKind, Operation

URI = "file:///notes.md"


def snapshot(version: int, *suggestions: SuggestionBody) -> DocumentBody:
    return DocumentBody(
        version=version, state=CheckState.DONE, suggestions=list(suggestions), unchecked=[]
    )


def typo(
    start: int, original: str = "teh", options: Sequence[str] = ("the", "tea")
) -> SuggestionBody:
    return SuggestionBody(
        id=f"sg_{start}",
        start=start,
        end=start + len(original),
        original=original,
        kind=ErrorKind.SPELLING,
        operation=Operation.REPLACEMENT,
        explanation="Misspelt.",
        options=[OptionBody(replacement=o) for o in options],
    )


class FakeService:
    """Versions each PUT, records calls, and streams events the test pushes."""

    def __init__(self) -> None:
        self.puts: list[str] = []
        self.closed: list[str] = []
        self.dismissed: list[tuple[str, str]] = []
        self.words: list[tuple[str, list[str]]] = []
        self.streams: dict[str, asyncio.Queue[DocumentBody | ServiceError | None]] = {}
        self.stream_opens = 0
        self.hold = asyncio.Event()
        self.hold.set()
        self.fail_puts = False

    async def put_document(self, uri: str, text: str) -> int:
        await self.hold.wait()
        if self.fail_puts:
            raise ServiceError("down")
        self.puts.append(text)
        return len(self.puts)

    async def close_document(self, uri: str) -> bool:
        self.closed.append(uri)
        return True

    async def dismiss(self, uri: str, suggestion_id: str) -> bool:
        self.dismissed.append((uri, suggestion_id))
        return True

    async def add_words(self, uri: str, words: Sequence[str]) -> list[str]:
        self.words.append((uri, list(words)))
        return list(words)

    async def events(self, uri: str) -> AsyncIterator[DocumentBody | None]:
        self.stream_opens += 1
        queue = self.streams.setdefault(uri, asyncio.Queue())
        while True:
            item = await queue.get()
            if isinstance(item, ServiceError):
                raise item
            yield item
            if item is None:
                return

    def push(self, item: DocumentBody | ServiceError | None) -> None:
        self.streams.setdefault(URI, asyncio.Queue()).put_nowait(item)


class Editor:
    """Records published diagnostics and log lines."""

    def __init__(self) -> None:
        self.published: list[tuple[str, list[Diagnostic]]] = []
        self.logs: list[str] = []

    async def publish(self, uri: str, diagnostics: list[Diagnostic]) -> None:
        self.published.append((uri, diagnostics))

    def log(self, message: str) -> None:
        self.logs.append(message)


def make(service: FakeService, editor: Editor) -> LspProxy:
    async def no_wait(seconds: float) -> None:
        await asyncio.sleep(0)

    contract: DocumentsService = service
    return LspProxy(contract, publish=editor.publish, log=editor.log, sleep=no_wait)


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def test_snapshots_become_diagnostics_for_the_text_the_editor_has() -> None:
    service, editor = FakeService(), Editor()
    proxy = make(service, editor)

    async def run() -> None:
        await proxy.opened(URI, "Read teh book.")
        await settle()
        service.push(snapshot(1, typo(5)))
        await settle()
        await proxy.shutdown()

    asyncio.run(run())

    [(uri, diagnostics), closing] = editor.published
    assert uri == URI
    [diagnostic] = diagnostics
    assert diagnostic.range == Range(Position(0, 5), Position(0, 8))
    assert diagnostic.message == "'teh' → 'the' or 'tea': Misspelt."
    assert closing == (URI, [])
    assert service.closed == [URI]


def test_changes_while_sending_collapse_to_the_latest_text() -> None:
    service, editor = FakeService(), Editor()
    proxy = make(service, editor)

    async def run() -> None:
        service.hold.clear()
        await proxy.opened(URI, "a")
        await settle()
        await proxy.changed(URI, "ab")
        await proxy.changed(URI, "abc")
        service.hold.set()
        await settle()
        await proxy.shutdown()

    asyncio.run(run())
    assert service.puts == ["a", "abc"]


def test_snapshots_for_other_versions_are_not_shown() -> None:
    service, editor = FakeService(), Editor()
    proxy = make(service, editor)

    async def run() -> None:
        await proxy.opened(URI, "Read teh book.")
        await settle()
        await proxy.changed(URI, "Now read teh book.")
        await settle()
        service.push(snapshot(1, typo(5)))
        await settle()
        assert editor.published == []
        service.push(snapshot(2, typo(9)))
        await settle()
        await proxy.shutdown()

    asyncio.run(run())
    [(_, [diagnostic]), _closing] = editor.published
    assert diagnostic.range.start == Position(0, 9)


def test_a_snapshot_that_arrives_before_its_version_is_shown_once_known() -> None:
    service, editor = FakeService(), Editor()
    proxy = make(service, editor)

    async def run() -> None:
        await proxy.opened(URI, "Read teh book.")
        await settle()
        service.hold.clear()
        await proxy.changed(URI, "Now read teh book.")
        service.push(snapshot(2, typo(9)))
        await settle()
        assert editor.published == []
        service.hold.set()
        await settle()
        await proxy.shutdown()

    asyncio.run(run())
    assert len(editor.published[0][1]) == 1


def test_a_lost_stream_resends_the_text_and_reconnects() -> None:
    service, editor = FakeService(), Editor()
    proxy = make(service, editor)

    async def run() -> None:
        await proxy.opened(URI, "Hi.")
        await settle()
        service.push(ServiceError("connection reset"))
        await settle()
        await proxy.shutdown()

    asyncio.run(run())
    assert service.puts == ["Hi.", "Hi."]
    assert service.stream_opens == 2
    assert any("lost events" in line for line in editor.logs)


def test_failed_sends_are_logged() -> None:
    service, editor = FakeService(), Editor()
    service.fail_puts = True
    proxy = make(service, editor)

    async def run() -> None:
        await proxy.opened(URI, "Hi.")
        await settle()

    asyncio.run(run())
    assert editor.logs == ["langcheck: could not send file:///notes.md: down"]


def test_code_actions_offer_options_dictionary_and_dismiss() -> None:
    service, editor = FakeService(), Editor()
    proxy = make(service, editor)

    async def run() -> list[object]:
        await proxy.opened(URI, "Read teh book. I go there.")
        await settle()
        service.push(snapshot(1, typo(5), typo(17, "go there", ("went there",))))
        await settle()
        inside_typo = Range(Position(0, 6), Position(0, 6))
        actions: list[object] = list(proxy.code_actions(URI, inside_typo))
        actions.append(proxy.code_actions(URI, Range(Position(0, 17), Position(0, 18))))
        actions.append(proxy.code_actions("file:///other.md", inside_typo))
        await proxy.dismiss(URI, "sg_5")
        await proxy.add_to_dictionary(URI, "teh")
        await proxy.shutdown()
        return actions

    *typo_actions, phrase_actions, other = asyncio.run(run())

    assert [a.title for a in typo_actions if isinstance(a, ReplaceAction | CommandAction)] == [
        "Replace with 'the'",
        "Replace with 'tea'",
        "Add 'teh' to the dictionary",
        "Dismiss",
    ]
    first = typo_actions[0]
    assert isinstance(first, ReplaceAction)
    assert first.preferred
    dismiss = typo_actions[3]
    assert isinstance(dismiss, CommandAction)
    assert (dismiss.command, dismiss.arguments) == (ProxyCommand.DISMISS, (URI, "sg_5"))
    assert isinstance(phrase_actions, list)
    assert [a.title for a in phrase_actions] == ["Replace with 'went there'", "Dismiss"]
    assert other == []
    assert service.dismissed == [(URI, "sg_5")]
    assert service.words == [(URI, ["teh"])]


def test_closing_an_unknown_document_does_nothing() -> None:
    service, editor = FakeService(), Editor()

    asyncio.run(make(service, editor).closed(URI))

    assert service.closed == []
    assert editor.published == []


def test_diagnostics_for_uses_utf16_columns_and_empty_options() -> None:
    [diagnostic] = diagnostics_for("😀 teh", snapshot(1, typo(2, "teh", ())))

    assert diagnostic.range == Range(Position(0, 3), Position(0, 6))
    assert diagnostic.message == "'teh': Misspelt."
    assert diagnostic.options == ()
