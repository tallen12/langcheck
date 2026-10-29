from langcheck.libreoffice.cache import DocumentCache
from langcheck.libreoffice.client import ServiceError
from langcheck.libreoffice.schema import DocumentState, Suggestion
from langcheck.libreoffice.worker import CheckWorker, DocumentsService

URI = "file:///a.odt"
TYPO = Suggestion("sg_5", 5, 8, "teh", "Spelling", "Misspelt.", ("the",))


class FakeService:
    """Versions each put; reports pending for ``pending_polls`` polls, then done."""

    def __init__(self, *, pending_polls: int = 1, fail: bool = False) -> None:
        self.puts: list[str] = []
        self.polls = 0
        self.pending_polls = pending_polls
        self.fail = fail

    def put_document(self, uri: str, text: str) -> int:
        if self.fail:
            raise ServiceError("down")
        self.puts.append(text)
        return len(self.puts)

    def document(self, uri: str) -> DocumentState:
        self.polls += 1
        settled = self.polls > self.pending_polls
        suggestions = (TYPO,) if "teh" in self.puts[-1] else ()
        return DocumentState(len(self.puts), "done" if settled else "checking", suggestions)


def make(
    service: FakeService, *, timeout: float = 60.0
) -> tuple[CheckWorker, DocumentCache, list[str], list[str]]:
    cache = DocumentCache()
    changed: list[str] = []
    logs: list[str] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    contract: DocumentsService = service
    worker = CheckWorker(
        contract,
        cache,
        on_changed=changed.append,
        log=logs.append,
        sleep=sleep,
        clock=lambda: now[0],
        poll_interval=1.0,
        timeout=timeout,
    )
    return worker, cache, changed, logs


def test_checks_the_text_and_reports_changed_suggestions() -> None:
    service = FakeService(pending_polls=2)
    worker, cache, changed, _ = make(service)

    worker.submit(URI, "Read teh book.")
    assert worker.run_once()
    assert not worker.run_once()

    assert changed == [URI]
    assert [e.start for e in cache.errors_for(URI, "Read teh book.")] == [5]
    assert service.polls == 3


def test_unchanged_text_is_not_sent_again_and_unchanged_results_not_reported() -> None:
    service = FakeService(pending_polls=0)
    worker, _, changed, _ = make(service)

    worker.submit(URI, "Read teh book.")
    worker.run_once()
    worker.submit(URI, "Read teh book.")
    assert not worker.run_once()
    worker.submit(URI, "Read teh book. More.")
    worker.run_once()

    assert service.puts == ["Read teh book.", "Read teh book. More."]
    assert changed == [URI]


def test_only_the_latest_text_is_queued() -> None:
    service = FakeService(pending_polls=0)
    worker, _, _, _ = make(service)

    worker.submit(URI, "one")
    worker.submit(URI, "two")
    worker.run_once()

    assert service.puts == ["two"]


def test_errors_and_timeouts_are_logged() -> None:
    worker, _, changed, logs = make(FakeService(fail=True))
    worker.submit(URI, "Hi.")
    worker.run_once()

    slow, _, slow_changed, slow_logs = make(FakeService(pending_polls=100), timeout=3)
    slow.submit(URI, "Hi.")
    slow.run_once()

    assert logs == ["langcheck: down"]
    assert slow_logs == [f"langcheck: gave up waiting for {URI}"]
    assert changed == slow_changed == []


def test_the_thread_processes_submissions_and_stops() -> None:
    service = FakeService(pending_polls=0)
    worker, _, _, _ = make(service)
    worker.start()
    worker.submit(URI, "Hi.")
    for _ in range(200):
        if service.puts:
            break
        import time

        time.sleep(0.005)
    worker.stop()

    assert service.puts == ["Hi."]
