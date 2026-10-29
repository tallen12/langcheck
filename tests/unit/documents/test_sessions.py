import asyncio
from collections.abc import Collection

import pytest

from langcheck.check.checker import CheckResult
from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.context.models import DocumentContext, Register
from langcheck.dictionary.memory_store import MemoryDictionaryStore
from langcheck.documents.events import Closed, DocumentEvent, Snapshot
from langcheck.documents.memory_store import MemoryDocumentStore
from langcheck.documents.models import CheckState, DocumentKey
from langcheck.documents.sessions import (
    Accepted,
    DocumentSessions,
    NotOpen,
    Updated,
    VersionConflict,
    changed_share,
)
from langcheck.limits.models import Admission, Permit, RateLimited
from langcheck.limits.user_limiter import Unlimited
from langcheck.llm.structured import Failed
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter

KEY = DocumentKey("ana", "file:///notes.md")


class TypoChecker:
    """Suggests "the" for every "teh" and records each check."""

    def __init__(self) -> None:
        self.checked: list[tuple[str, str, DocumentContext]] = []
        self.words: list[frozenset[str]] = []
        self.fail = False

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        self.checked.append((text, user, context))
        self.words.append(frozenset(dictionary))
        if self.fail:
            msg = "checker broke"
            raise RuntimeError(msg)
        found: list[Suggestion] = []
        start = text.find("teh")
        while start != -1:
            span = Span(start, start + 3)
            found.append(
                Suggestion(
                    id=suggestion_id(span, "teh", ErrorKind.SPELLING),
                    span=span,
                    original="teh",
                    kind=ErrorKind.SPELLING,
                    operation=Operation.REPLACEMENT,
                    explanation="",
                    options=(Option("the"), Option("tea")),
                )
            )
            start = text.find("teh", start + 1)
        return CheckResult(tuple(found), (), ())


class CountingContexts:
    """Builds a context naming how many times it was built; can fail."""

    def __init__(self, *, fail: bool = False) -> None:
        self.built: list[str] = []
        self.fail = fail

    async def build(
        self, text: str, *, dictionary: Collection[str] = ()
    ) -> DocumentContext | Failed:
        self.built.append(text)
        if self.fail:
            return Failed("no reply")
        return DocumentContext(f"build {len(self.built)}", Register.NEUTRAL, "", (), "")


class Gate:
    """A sleep that waits until opened, to hold checks in their debounce."""

    def __init__(self) -> None:
        self.opened = asyncio.Event()
        self.waits = 0

    async def sleep(self, seconds: float) -> None:
        self.waits += 1
        await self.opened.wait()


def make_sessions(
    checker: TypoChecker | None = None,
    contexts: CountingContexts | None = None,
    *,
    gate: Gate | None = None,
    refresh_share: float = 0.2,
    dictionary: MemoryDictionaryStore | None = None,
) -> DocumentSessions:
    async def no_wait(seconds: float) -> None:
        await asyncio.sleep(0)

    return DocumentSessions(
        MemoryDocumentStore(),
        checker=checker or TypoChecker(),
        contexts=contexts or CountingContexts(),
        splitter=RegexSentenceSplitter(),
        limiter=Unlimited(),
        dictionary=dictionary or MemoryDictionaryStore(),
        debounce=0.75,
        refresh_share=refresh_share,
        sleep=gate.sleep if gate is not None else no_wait,
    )


def test_opening_a_document_checks_it_in_the_background() -> None:
    checker = TypoChecker()
    sessions = make_sessions(checker)

    async def run() -> None:
        assert await sessions.update(KEY, "Read teh book.") == Updated(1)
        pending = await sessions.get(KEY)
        assert pending is not None
        assert pending.state is CheckState.PENDING
        await sessions.wait_idle()
        done = await sessions.get(KEY)
        assert done is not None
        assert done.state is CheckState.DONE
        assert [s.span for s in done.suggestions] == [Span(5, 8)]
        assert done.context is not None
        assert done.context.context.summary == "build 1"

    asyncio.run(run())
    assert checker.checked[0][:2] == ("Read teh book.", "ana")


def test_updates_move_suggestions_at_once_and_newer_versions_cancel_older_checks() -> None:
    checker = TypoChecker()
    gate = Gate()
    sessions = make_sessions(checker, gate=gate)

    async def run() -> None:
        await sessions.update(KEY, "Read teh book.")
        gate.opened.set()
        await sessions.wait_idle()
        gate.opened.clear()
        assert await sessions.update(KEY, "Hi. Read teh book.", base_version=1) == Updated(2)
        assert await sessions.update(KEY, "Hi there. Read teh book.") == Updated(3)
        moved = await sessions.get(KEY)
        assert moved is not None
        assert [s.span for s in moved.suggestions] == [Span(15, 18)]
        gate.opened.set()
        await sessions.wait_idle()

    asyncio.run(run())
    assert [text for text, _, _ in checker.checked] == [
        "Read teh book.",
        "Hi there. Read teh book.",
    ]


def test_a_stale_base_version_is_a_conflict() -> None:
    sessions = make_sessions()

    async def run() -> None:
        await sessions.update(KEY, "One.")
        assert await sessions.update(KEY, "Two.", base_version=7) == VersionConflict(1)
        await sessions.aclose()

    asyncio.run(run())


def test_accept_applies_the_option_as_a_new_version() -> None:
    sessions = make_sessions()

    async def run() -> None:
        await sessions.update(KEY, "Read teh book.")
        await sessions.wait_idle()
        document = await sessions.get(KEY)
        assert document is not None
        [typo] = document.suggestions
        result = await sessions.accept(KEY, typo.id, option=1, base_version=1)
        assert result == Accepted(2, Span(5, 8), "tea")
        assert await sessions.accept(KEY, typo.id, option=0, base_version=1) == VersionConflict(2)
        assert await sessions.accept(KEY, "sg_missing", option=0, base_version=None) == NotOpen()
        await sessions.wait_idle()
        accepted = await sessions.get(KEY)
        assert accepted is not None
        assert accepted.text == "Read tea book."
        assert accepted.suggestions == ()

    asyncio.run(run())


def test_dismissed_issues_stay_dismissed_after_edits_elsewhere() -> None:
    sessions = make_sessions()

    async def run() -> None:
        await sessions.update(KEY, "Read teh book.")
        await sessions.wait_idle()
        document = await sessions.get(KEY)
        assert document is not None
        assert await sessions.dismiss(KEY, document.suggestions[0].id)
        assert not await sessions.dismiss(KEY, "sg_missing")
        await sessions.update(KEY, "Hello. Read teh book. Then teh end.")
        await sessions.wait_idle()
        edited = await sessions.get(KEY)
        assert edited is not None
        assert [s.span.start for s in edited.suggestions] == [27]

    asyncio.run(run())


def test_watchers_get_snapshots_then_closed() -> None:
    sessions = make_sessions()

    async def run() -> list[DocumentEvent]:
        await sessions.update(KEY, "Read teh book.")
        await sessions.wait_idle()
        events: list[DocumentEvent] = []
        async with sessions.watch(KEY) as queue:
            await sessions.update(KEY, "Read teh book!")
            await sessions.wait_idle()
            assert await sessions.close(KEY)
            while not queue.empty():
                events.append(queue.get_nowait())
        return events

    events = asyncio.run(run())

    snapshots = [e for e in events if isinstance(e, Snapshot)]
    assert [(s.version, s.state) for s in snapshots] == [
        (1, CheckState.DONE),
        (2, CheckState.PENDING),
        (2, CheckState.CHECKING),
        (2, CheckState.DONE),
    ]
    assert events[-1] == Closed()


def test_context_is_rebuilt_only_when_enough_changed() -> None:
    contexts = CountingContexts()
    sessions = make_sessions(contexts=contexts, refresh_share=0.5)

    async def run() -> None:
        await sessions.update(KEY, "One. Two. Three. Four.")
        await sessions.wait_idle()
        await sessions.update(KEY, "One. Two. Three. Five.")
        await sessions.wait_idle()
        await sessions.update(KEY, "One. Six. Seven. Eight.")
        await sessions.wait_idle()

    asyncio.run(run())
    assert len(contexts.built) == 2


def test_failed_context_builds_fall_back_to_an_empty_context() -> None:
    checker = TypoChecker()
    sessions = make_sessions(checker, CountingContexts(fail=True))

    async def run() -> None:
        await sessions.update(KEY, "Read teh book.")
        await sessions.wait_idle()

    asyncio.run(run())
    assert checker.checked[0][2] == DocumentContext.empty()


def test_a_broken_checker_marks_the_version_failed(caplog: pytest.LogCaptureFixture) -> None:
    checker = TypoChecker()
    checker.fail = True
    sessions = make_sessions(checker)

    async def run() -> CheckState | None:
        await sessions.update(KEY, "Read teh book.")
        await sessions.wait_idle()
        document = await sessions.get(KEY)
        return document.state if document is not None else None

    assert asyncio.run(run()) is CheckState.FAILED
    assert "check of file:///notes.md failed" in caplog.text


def test_rename_moves_the_document_and_refuses_open_targets() -> None:
    sessions = make_sessions()
    other = DocumentKey("ana", "file:///other.md")

    async def run() -> None:
        await sessions.update(KEY, "Read teh book.")
        await sessions.update(other, "Other.")
        await sessions.wait_idle()
        assert await sessions.rename(KEY, other.uri) == VersionConflict(1)
        assert await sessions.rename(KEY, "file:///renamed.md") == Updated(1)
        assert await sessions.get(KEY) is None
        renamed = await sessions.get(DocumentKey("ana", "file:///renamed.md"))
        assert renamed is not None
        assert renamed.text == "Read teh book."
        assert await sessions.rename(KEY, "file:///x.md") == NotOpen()
        assert not await sessions.close(KEY)

    asyncio.run(run())


def test_documents_are_scoped_by_user() -> None:
    sessions = make_sessions()

    async def run() -> None:
        await sessions.update(KEY, "Ana's text.")
        assert await sessions.get(DocumentKey("ben", KEY.uri)) is None
        await sessions.aclose()

    asyncio.run(run())


def test_changed_share_counts_repeats() -> None:
    assert changed_share(("a", "b"), ()) == 0.0
    assert changed_share(("a", "b"), ("a", "b", "c", "d")) == 0.5
    assert changed_share(("a",), ("a", "a")) == 0.5


class LimitedOnce:
    """Refuses the first check, then admits."""

    def __init__(self) -> None:
        self.asked = 0

    def admit(self, user: str) -> Admission:
        self.asked += 1
        return RateLimited(5.0) if self.asked == 1 else Permit(lambda: None)


def test_rate_limited_checks_wait_and_retry() -> None:
    checker = TypoChecker()
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        await asyncio.sleep(0)

    sessions = DocumentSessions(
        MemoryDocumentStore(),
        checker=checker,
        contexts=CountingContexts(),
        splitter=RegexSentenceSplitter(),
        limiter=LimitedOnce(),
        dictionary=MemoryDictionaryStore(),
        debounce=0.5,
        sleep=sleep,
    )

    async def run() -> list[CheckState]:
        states: list[CheckState] = []
        async with sessions.watch(KEY) as queue:
            await sessions.update(KEY, "Read teh book.")
            await sessions.wait_idle()
            while not queue.empty():
                event = queue.get_nowait()
                if isinstance(event, Snapshot):
                    states.append(event.state)
        return states

    states = asyncio.run(run())

    assert waits == [0.5, 5.0]
    assert states == [
        CheckState.PENDING,
        CheckState.RATE_LIMITED,
        CheckState.CHECKING,
        CheckState.DONE,
    ]
    assert len(checker.checked) == 1


def test_session_checks_use_the_documents_dictionary() -> None:
    checker = TypoChecker()
    dictionary = MemoryDictionaryStore()
    dictionary.add("ana", KEY.uri, ["Shire"])
    sessions = make_sessions(checker, dictionary=dictionary)

    async def run() -> bool:
        await sessions.update(KEY, "Read teh book.")
        await sessions.wait_idle()
        rechecked = await sessions.dictionary_changed(KEY)
        await sessions.wait_idle()
        return rechecked and not await sessions.dictionary_changed(DocumentKey("ana", "x"))

    assert asyncio.run(run())
    assert len(checker.checked) == 2
    assert checker.words == [frozenset({"Shire"})] * 2
