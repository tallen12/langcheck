"""Sending documents to the service in a background thread.

LibreOffice must get an answer from ``doProofreading`` at once, so the full text goes to
the service from this thread. Only the latest text of each document is kept: edits made
while a check runs replace the queued text instead of queueing more work.
"""

import threading
import time
from collections.abc import Callable
from typing import Protocol

from langcheck.libreoffice.cache import Checked, DocumentCache
from langcheck.libreoffice.client import ServiceError
from langcheck.libreoffice.schema import DocumentState

DEFAULT_POLL_INTERVAL = 0.5
DEFAULT_TIMEOUT = 120.0


class DocumentsService(Protocol):
    """The calls the worker makes (``ServiceClient`` has them)."""

    def put_document(self, uri: str, text: str) -> int:
        """Open or update a document; its version."""
        ...

    def document(self, uri: str) -> DocumentState:
        """The document's current suggestions and check state."""
        ...


class CheckWorker:
    """Checks the latest submitted text of each document, one at a time.

    ``on_changed(uri)`` is called from the worker thread whenever a document's
    suggestions change, so the proofreader can ask LibreOffice to re-check.
    """

    def __init__(
        self,
        service: DocumentsService,
        cache: DocumentCache,
        *,
        on_changed: Callable[[str], None],
        log: Callable[[str], None],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._service = service
        self._cache = cache
        self._on_changed = on_changed
        self._log = log
        self._sleep = sleep
        self._clock = clock
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._condition = threading.Condition()
        self._pending: dict[str, str] = {}
        self._sent: dict[str, str] = {}
        self._stopped = False
        self._thread: threading.Thread | None = None

    def submit(self, uri: str, text: str) -> None:
        """Queue ``text`` for ``uri`` unless it is what was last checked."""
        with self._condition:
            if self._sent.get(uri) == text and uri not in self._pending:
                return
            self._pending[uri] = text
            self._condition.notify()

    def start(self) -> None:
        """Start the worker thread (a daemon, so it never holds LibreOffice open)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="langcheck", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Ask the thread to finish its current request and exit."""
        with self._condition:
            self._stopped = True
            self._condition.notify()

    def run_once(self) -> bool:
        """Check one queued document now, on this thread; ``False`` if none is queued."""
        with self._condition:
            if not self._pending:
                return False
            uri = next(iter(self._pending))
            text = self._pending.pop(uri)
        self._check(uri, text)
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
            self.run_once()

    def _check(self, uri: str, text: str) -> None:
        try:
            version = self._service.put_document(uri, text)
            with self._condition:
                self._sent[uri] = text
            state = self._settled(uri, version)
        except ServiceError as error:
            self._log(f"langcheck: {error}")
            return
        if state is None or state.version != version:
            return
        if self._cache.store(uri, Checked(text, state.suggestions)):
            self._on_changed(uri)

    def _settled(self, uri: str, version: int) -> DocumentState | None:
        """Poll until ``version`` is checked; ``None`` if newer text arrives or it takes
        too long.
        """
        deadline = self._clock() + self._timeout
        while self._clock() < deadline:
            state = self._service.document(uri)
            if state.version != version or state.settled:
                return state
            with self._condition:
                if uri in self._pending or self._stopped:
                    return None
            self._sleep(self._poll_interval)
        self._log(f"langcheck: gave up waiting for {uri}")
        return None
