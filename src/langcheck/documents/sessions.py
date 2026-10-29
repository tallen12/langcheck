"""Open documents: updates, background checks, accepting and dismissing suggestions.

An update stores the new version at once, moving suggestions from unchanged sentences,
then checks the whole text after a short debounce, waiting out the user's rate limit.
The result cache means only new or edited sentences reach the LLM. A newer version
cancels the pending check of an older one, and a check whose version is no longer
current is discarded.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace

from langcheck.check.apply import Applied, NoSuchOption, Stale, apply_option
from langcheck.check.protocols import TextChecker
from langcheck.context.models import DocumentContext
from langcheck.context.protocols import ContextSource
from langcheck.dictionary.protocols import DictionaryStore
from langcheck.documents.events import Closed, DocumentEvent, Snapshot
from langcheck.documents.models import (
    BuiltContext,
    CheckState,
    Dismissal,
    Document,
    DocumentKey,
)
from langcheck.documents.protocols import DocumentStore
from langcheck.limits.models import Permit, RateLimited
from langcheck.limits.protocols import CheckLimiter
from langcheck.text.diff import SentenceDiff, diff_sentences
from langcheck.text.models import Span
from langcheck.text.protocols import SentenceSplitter
from langcheck.text.segment import segment

DEFAULT_DEBOUNCE = 0.75
DEFAULT_REFRESH_SHARE = 0.2

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Updated:
    """The document now has ``version``."""

    version: int


@dataclass(frozen=True, slots=True)
class VersionConflict:
    """The client's base version is not the current one; it should send the full text
    of the ``current`` version's successor or re-open the document.
    """

    current: int


@dataclass(frozen=True, slots=True)
class NotOpen:
    """No document is open at that key (or no suggestion has that ID)."""


@dataclass(frozen=True, slots=True)
class Accepted:
    """The option was applied: ``span`` of the old version now reads ``replacement``."""

    version: int
    span: Span
    replacement: str


type UpdateResult = Updated | VersionConflict
type AcceptResult = Accepted | VersionConflict | NotOpen


class DocumentSessions:
    """Keeps every open document and checks each new version in the background."""

    def __init__(
        self,
        store: DocumentStore,
        *,
        checker: TextChecker,
        contexts: ContextSource,
        splitter: SentenceSplitter,
        limiter: CheckLimiter,
        dictionary: DictionaryStore,
        debounce: float = DEFAULT_DEBOUNCE,
        refresh_share: float = DEFAULT_REFRESH_SHARE,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._store = store
        self._checker = checker
        self._contexts = contexts
        self._splitter = splitter
        self._limiter = limiter
        self._dictionary = dictionary
        self._debounce = debounce
        self._refresh_share = refresh_share
        self._sleep = sleep
        self._locks: dict[DocumentKey, asyncio.Lock] = {}
        self._checks: dict[DocumentKey, asyncio.Task[None]] = {}
        self._watchers: dict[DocumentKey, set[asyncio.Queue[DocumentEvent]]] = {}

    async def get(self, key: DocumentKey) -> Document | None:
        """The current version of the document at ``key``."""
        return await self._store.get(key)

    async def update(
        self, key: DocumentKey, text: str, *, base_version: int | None = None
    ) -> UpdateResult:
        """Open the document or store a new version of it, and schedule its check.

        With ``base_version``, the update applies only if that is the current version;
        without it, the text replaces whatever is there.
        """
        async with self._lock(key):
            current = await self._store.get(key)
            if current is None:
                document = Document(key=key, text=text, version=1, state=CheckState.PENDING)
            elif base_version is not None and base_version != current.version:
                return VersionConflict(current.version)
            else:
                document = current.edited(text, self._diff(current.text, text))
            await self._store.put(document)
        self._publish(key, Snapshot.of(document))
        self._schedule(key, document.version)
        return Updated(document.version)

    async def accept(
        self, key: DocumentKey, suggestion_id: str, *, option: int, base_version: int | None
    ) -> AcceptResult:
        """Apply an option of a current suggestion as a new version of the document."""
        async with self._lock(key):
            current = await self._store.get(key)
            if current is None:
                return NotOpen()
            if base_version is not None and base_version != current.version:
                return VersionConflict(current.version)
            suggestion = current.suggestion(suggestion_id)
            if suggestion is None:
                return NotOpen()
            match apply_option(current.text, suggestion, option=option):
                case Applied(text=text):
                    replacement = suggestion.options[option].replacement
                case Stale() | NoSuchOption():
                    return NotOpen()
            document = current.edited(text, self._diff(current.text, text))
            await self._store.put(document)
        self._publish(key, Snapshot.of(document))
        self._schedule(key, document.version)
        return Accepted(document.version, suggestion.span, replacement)

    async def dismiss(self, key: DocumentKey, suggestion_id: str) -> bool:
        """Hide a suggestion and stop the same issue being suggested in this session."""
        async with self._lock(key):
            current = await self._store.get(key)
            suggestion = current.suggestion(suggestion_id) if current is not None else None
            if current is None or suggestion is None:
                return False
            document = replace(
                current,
                suggestions=tuple(s for s in current.suggestions if s.id != suggestion_id),
                dismissals=(*current.dismissals, Dismissal(suggestion.span, suggestion.original)),
            )
            await self._store.put(document)
        self._publish(key, Snapshot.of(document))
        return True

    async def dictionary_changed(self, key: DocumentKey) -> bool:
        """Re-check the open document after its dictionary changed. Every sentence comes
        from the cache, so only the dictionary filter runs again. ``False`` if not open.
        """
        async with self._lock(key):
            current = await self._store.get(key)
            if current is None:
                return False
            pending = replace(current, state=CheckState.PENDING)
            await self._store.put(pending)
        self._publish(key, Snapshot.of(pending))
        self._schedule(key, pending.version)
        return True

    async def close(self, key: DocumentKey) -> bool:
        """Close the document, cancelling its check; ``False`` when it was not open."""
        self._cancel(key)
        async with self._lock(key):
            closed = await self._store.delete(key)
        self._publish(key, Closed())
        self._watchers.pop(key, None)
        self._locks.pop(key, None)
        return closed

    async def rename(self, key: DocumentKey, new_uri: str) -> UpdateResult | NotOpen:
        """Move the document to ``new_uri`` (same user); watchers of the old key are
        told it closed. Renaming onto an open document is a conflict.
        """
        target = DocumentKey(key.user, new_uri)
        async with self._lock(key), self._lock(target):
            current = await self._store.get(key)
            if current is None:
                return NotOpen()
            existing = await self._store.get(target)
            if existing is not None:
                return VersionConflict(existing.version)
            await self._store.delete(key)
            await self._store.put(replace(current, key=target))
            await asyncio.to_thread(self._dictionary.rename, key.user, key.uri, new_uri)
        self._cancel(key)
        self._publish(key, Closed())
        self._watchers.pop(key, None)
        if current.state in (CheckState.PENDING, CheckState.CHECKING):
            self._schedule(target, current.version)
        return Updated(current.version)

    @asynccontextmanager
    async def watch(self, key: DocumentKey) -> AsyncGenerator[asyncio.Queue[DocumentEvent]]:
        """A queue of events for ``key``, starting with the current snapshot if open."""
        queue: asyncio.Queue[DocumentEvent] = asyncio.Queue()
        self._watchers.setdefault(key, set()).add(queue)
        current = await self._store.get(key)
        if current is not None:
            queue.put_nowait(Snapshot.of(current))
        try:
            yield queue
        finally:
            watchers = self._watchers.get(key)
            if watchers is not None:
                watchers.discard(queue)

    async def wait_idle(self) -> None:
        """Wait until no check is pending or running, for tests and shutdown."""
        while self._checks:
            await asyncio.gather(*self._checks.values(), return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel every pending check."""
        for key in list(self._checks):
            self._cancel(key)
        await self.wait_idle()

    def _lock(self, key: DocumentKey) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def _diff(self, old: str, new: str) -> SentenceDiff:
        return diff_sentences(
            segment(old, splitter=self._splitter), segment(new, splitter=self._splitter)
        )

    def _publish(self, key: DocumentKey, event: DocumentEvent) -> None:
        for queue in self._watchers.get(key, ()):
            queue.put_nowait(event)

    def _cancel(self, key: DocumentKey) -> None:
        task = self._checks.pop(key, None)
        if task is not None:
            task.cancel()

    def _schedule(self, key: DocumentKey, version: int) -> None:
        self._cancel(key)
        task = asyncio.create_task(self._check(key, version))
        self._checks[key] = task
        task.add_done_callback(lambda done: self._finished(key, done))

    def _finished(self, key: DocumentKey, task: asyncio.Task[None]) -> None:
        if self._checks.get(key) is task:
            del self._checks[key]
        if not task.cancelled() and task.exception() is not None:
            _log.error("check of %s failed", key.uri, exc_info=task.exception())

    async def _check(self, key: DocumentKey, version: int) -> None:
        await self._sleep(self._debounce)
        while True:
            match self._limiter.admit(key.user):
                case Permit() as permit:
                    break
                case RateLimited(retry_after=wait):
                    if await self._set_state(key, version, CheckState.RATE_LIMITED) is None:
                        return
                    await self._sleep(wait)
        with permit:
            document = await self._set_state(key, version, CheckState.CHECKING)
            if document is None:
                return
            try:
                words = await asyncio.to_thread(self._dictionary.words, key.user, key.uri)
                built = await self._context_for(document, words)
                result = await self._checker.check(
                    document.text, user=key.user, context=built.context, dictionary=words
                )
            except Exception:
                await self._set_state(key, version, CheckState.FAILED)
                raise
        async with self._lock(key):
            current = await self._store.get(key)
            if current is None or current.version != version:
                return
            checked = replace(
                current,
                state=CheckState.DONE,
                suggestions=tuple(s for s in result.suggestions if not current.is_dismissed(s)),
                unchecked=result.unchecked,
                context=built,
            )
            await self._store.put(checked)
        self._publish(key, Snapshot.of(checked))

    async def _set_state(
        self, key: DocumentKey, version: int, state: CheckState
    ) -> Document | None:
        async with self._lock(key):
            current = await self._store.get(key)
            if current is None or current.version != version:
                return None
            document = replace(current, state=state)
            await self._store.put(document)
        self._publish(key, Snapshot.of(document))
        return document

    async def _context_for(self, document: Document, words: frozenset[str]) -> BuiltContext:
        """The document's context, rebuilt when enough of the text changed since."""
        hashes = tuple(s.hash for s in segment(document.text, splitter=self._splitter).sentences)
        built = document.context
        if (
            built is not None
            and changed_share(built.sentence_hashes, hashes) <= self._refresh_share
        ):
            return built
        fresh = await self._contexts.build(document.text, dictionary=words)
        if isinstance(fresh, DocumentContext):
            return BuiltContext(fresh, hashes)
        # Keep checking with the old context (or none) rather than failing the check.
        return built if built is not None else BuiltContext(DocumentContext.empty(), ())


def changed_share(before: tuple[str, ...], after: tuple[str, ...]) -> float:
    """The share of ``after``'s sentence hashes not found in ``before`` (as multisets)."""
    if not after:
        return 0.0
    remaining = Counter(before)
    unchanged = 0
    for value in after:
        if remaining[value] > 0:
            remaining[value] -= 1
            unchanged += 1
    return 1 - unchanged / len(after)
