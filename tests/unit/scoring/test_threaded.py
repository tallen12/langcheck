import asyncio
import threading
from collections.abc import Sequence

import pytest

from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import AsyncSentenceScorer
from langcheck.scoring.threaded import ThreadedScorer
from langcheck.text.models import Span


def text(value: str) -> ScoredText:
    return ScoredText(value, Span(0, 0))


class GatedScorer:
    """Scores by length, records each batch and its thread, and can hold the first batch
    until released so later requests queue up behind it.
    """

    def __init__(self, *, hold_first: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.threads: set[str] = set()
        self.release = threading.Event()
        if not hold_first:
            self.release.set()

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        self.threads.add(threading.current_thread().name)
        self.batches.append([t.text for t in texts])
        self.release.wait(timeout=5)
        return [float(len(t.text)) for t in texts]


class FailingScorer:
    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        msg = "out of memory"
        raise RuntimeError(msg)


def test_scores_in_one_worker_thread() -> None:
    scorer = GatedScorer()

    async def run() -> list[float]:
        async with ThreadedScorer(scorer) as threaded:
            contract: AsyncSentenceScorer = threaded
            return await contract.score([text("ab"), text("abc")])

    assert asyncio.run(run()) == [2.0, 3.0]
    [thread] = scorer.threads
    assert thread.startswith("langcheck-scorer")
    assert thread != threading.current_thread().name


def test_requests_waiting_behind_a_busy_model_share_a_batch_in_order() -> None:
    scorer = GatedScorer(hold_first=True)

    async def run() -> list[list[float]]:
        async with ThreadedScorer(scorer, max_batch=3) as threaded:
            first = asyncio.create_task(threaded.score([text("a")]))
            await asyncio.sleep(0.05)
            waiting = [
                asyncio.create_task(threaded.score([text("bb")])),
                asyncio.create_task(threaded.score([text("ccc"), text("dddd")])),
                asyncio.create_task(threaded.score([text("eeeee")])),
            ]
            await asyncio.sleep(0.05)
            scorer.release.set()
            return [await first, *[await task for task in waiting]]

    assert asyncio.run(run()) == [[1.0], [2.0], [3.0, 4.0], [5.0]]
    assert scorer.batches == [["a"], ["bb", "ccc", "dddd"], ["eeeee"]]


def test_scorer_errors_reach_every_caller_in_the_batch() -> None:
    async def run() -> None:
        async with ThreadedScorer(FailingScorer()) as threaded:
            with pytest.raises(RuntimeError, match="out of memory"):
                await threaded.score([text("a")])
            with pytest.raises(RuntimeError, match="out of memory"):
                await threaded.score([text("b")])

    asyncio.run(run())


def test_stopping_fails_requests_in_flight_and_queued() -> None:
    scorer = GatedScorer(hold_first=True)

    async def run() -> list[BaseException | list[float]]:
        threaded = ThreadedScorer(scorer)
        async with threaded:
            in_flight = asyncio.create_task(threaded.score([text("a")]))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(threaded.score([text("b")]))
            await asyncio.sleep(0.05)
            scorer.release.set()
        return list(await asyncio.gather(in_flight, queued, return_exceptions=True))

    results = asyncio.run(asyncio.wait_for(run(), timeout=5))

    assert [type(r) for r in results] == [RuntimeError, RuntimeError]
    assert all("stopped" in str(r) for r in results)


def test_needs_to_be_running_and_skips_empty_requests() -> None:
    threaded = ThreadedScorer(GatedScorer())

    with pytest.raises(RuntimeError, match="not running"):
        asyncio.run(threaded.score([text("a")]))

    async def run() -> list[float]:
        async with threaded:
            return await threaded.score([])

    assert asyncio.run(run()) == []


def test_max_batch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        ThreadedScorer(GatedScorer(), max_batch=0)
