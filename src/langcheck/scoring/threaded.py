"""Running a blocking scorer in one worker thread behind a small batching queue."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import SentenceScorer

DEFAULT_MAX_BATCH = 256


@dataclass(frozen=True, slots=True)
class _Request:
    texts: tuple[ScoredText, ...]
    reply: asyncio.Future[list[float]]


class ThreadedScorer:
    """An ``AsyncSentenceScorer`` that runs ``scorer`` in a single worker thread.

    Requests that arrive while the model is busy are merged into one batch of at most
    ``max_batch`` texts (a larger request runs alone), so concurrent checks share forward
    passes and the event loop never blocks. Use it as an async context manager: the
    worker starts on entry and the thread is shut down on exit.
    """

    def __init__(self, scorer: SentenceScorer, *, max_batch: int = DEFAULT_MAX_BATCH) -> None:
        if max_batch < 1:
            msg = f"max_batch must be positive, got {max_batch}"
            raise ValueError(msg)
        self._scorer = scorer
        self._max_batch = max_batch
        self._queue: asyncio.Queue[_Request] = asyncio.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._worker: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langcheck-scorer")
        self._worker = asyncio.create_task(self._work())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        # Fail what the worker never reached, so no caller waits forever.
        while not self._queue.empty():
            _fail(self._queue.get_nowait(), _stopped())
        if self._executor is not None:
            await asyncio.to_thread(self._executor.shutdown, wait=True)
        self._worker = None
        self._executor = None

    async def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """Scores for ``texts``, computed in the worker thread."""
        if self._worker is None:
            msg = "ThreadedScorer is not running; use it as an async context manager"
            raise RuntimeError(msg)
        if not texts:
            return []
        reply = asyncio.Future[list[float]]()
        self._queue.put_nowait(_Request(tuple(texts), reply))
        return await reply

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        # A request that did not fit the last batch starts the next one, keeping order.
        carried: list[_Request] = []
        batch: list[_Request] = []
        try:
            while True:
                batch = [carried.pop() if carried else await self._queue.get()]
                size = len(batch[0].texts)
                while not self._queue.empty():
                    waiting = self._queue.get_nowait()
                    if size + len(waiting.texts) > self._max_batch:
                        carried.append(waiting)
                        break
                    batch.append(waiting)
                    size += len(waiting.texts)
                texts = [text for request in batch for text in request.texts]
                try:
                    scores = await loop.run_in_executor(self._executor, self._scorer.score, texts)
                except Exception as error:  # handed to every waiting caller, never dropped
                    for request in batch:
                        _fail(request, error)
                    continue
                start = 0
                for request in batch:
                    end = start + len(request.texts)
                    if not request.reply.done():
                        request.reply.set_result(scores[start:end])
                    start = end
        finally:
            # Stopped mid-batch: the requests taken off the queue get an answer too.
            for request in [*batch, *carried]:
                _fail(request, _stopped())


def _fail(request: _Request, error: BaseException) -> None:
    if not request.reply.done():
        request.reply.set_exception(error)


def _stopped() -> RuntimeError:
    return RuntimeError("ThreadedScorer stopped before scoring the request")
