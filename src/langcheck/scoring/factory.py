"""Building the option ranker from a scorer configuration.

The model adapters import PyTorch, so they are imported only when a configuration names
them; without a scorer the service runs without the PyTorch extras installed.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import assert_never

from langcheck.check.protocols import OptionRanker
from langcheck.scoring.combined import SumScorer
from langcheck.scoring.policy import ScoringPolicy
from langcheck.scoring.protocols import SentenceScorer
from langcheck.scoring.ranker import KeepOrder, ScoringRanker
from langcheck.scoring.threaded import DEFAULT_MAX_BATCH, ThreadedScorer


class ScorerAdapter(StrEnum):
    """The sentence scorers langcheck can load."""

    PLL = "pll"
    COLA = "cola"


@dataclass(frozen=True, slots=True)
class ScorerConfig:
    """Which scorers to load, where, and the filtering policy. No adapters: no scorer."""

    adapters: tuple[ScorerAdapter, ...] = ()
    device: str = "cpu"
    pll_model: str = "distilroberta-base"
    cola_model: str = "textattack/roberta-base-CoLA"
    window: int = 3
    max_batch: int = DEFAULT_MAX_BATCH
    policy: ScoringPolicy = field(default_factory=ScoringPolicy)

    @property
    def name(self) -> str:
        """Identifies everything that changes the ranking, for result cache keys."""
        models = {
            ScorerAdapter.PLL: f"{self.pll_model}@{self.window}",
            ScorerAdapter.COLA: self.cola_model,
        }
        policy = ",".join(
            f"{kind.value}:{rule.margin}"
            for kind, rule in sorted(self.policy.kinds.items())
            if rule.filter
        )
        return "+".join(f"{a.value}={models[a]}" for a in self.adapters) + f"|filter={policy}"


class ScorerUnavailableError(Exception):
    """A scorer is configured but PyTorch or transformers is not installed."""


type AdapterLoader = Callable[[ScorerAdapter, ScorerConfig], SentenceScorer]


def load_adapter(adapter: ScorerAdapter, config: ScorerConfig) -> SentenceScorer:
    """Load one scorer's model (blocking; downloads on first use)."""
    try:
        match adapter:
            case ScorerAdapter.PLL:
                from langcheck.scoring.pll import load_pll_scorer

                return load_pll_scorer(config.pll_model, device=config.device, window=config.window)
            case ScorerAdapter.COLA:
                from langcheck.scoring.cola import load_acceptability_scorer

                return load_acceptability_scorer(config.cola_model, device=config.device)
            case unreachable:
                assert_never(unreachable)
    except ImportError as error:
        msg = (
            f"the {adapter.value} scorer needs PyTorch and transformers: "
            "install a build with `just setup cpu` (or cuda, rocm), "
            "or turn scoring off with an empty scorer.adapters"
        )
        raise ScorerUnavailableError(msg) from error


@asynccontextmanager
async def open_ranker(
    config: ScorerConfig, *, load: AdapterLoader = load_adapter
) -> AsyncGenerator[OptionRanker]:
    """The ranker for ``config``, with its scorer thread running for the block.

    Without adapters, options keep the LLM's order. Models load off the event loop.
    """
    if not config.adapters:
        yield KeepOrder()
        return
    scorers = [await asyncio.to_thread(load, adapter, config) for adapter in config.adapters]
    scorer = scorers[0] if len(scorers) == 1 else SumScorer(scorers)
    async with ThreadedScorer(scorer, max_batch=config.max_batch) as threaded:
        yield ScoringRanker(threaded, policy=config.policy, name=config.name)
