import asyncio
import sys

import pytest

from langcheck.check.protocols import OptionRanker
from langcheck.check.suggestion import Option, Scored, Suggestion
from langcheck.scoring.factory import (
    ScorerAdapter,
    ScorerConfig,
    ScorerUnavailableError,
    load_adapter,
    open_ranker,
)
from langcheck.scoring.policy import KindPolicy, ScoringPolicy
from langcheck.scoring.protocols import SentenceScorer
from langcheck.scoring.ranker import KEEP_ORDER
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Sentence, Span
from tests.conftest import SyncTableScorer

SENTENCE = Sentence("S1", Span(0, 14), "Read teh book.")
TYPO = Suggestion(
    id="sg_1",
    span=Span(5, 8),
    original="teh",
    kind=ErrorKind.SPELLING,
    operation=Operation.REPLACEMENT,
    explanation="",
    options=(Option("the"),),
)


def test_without_adapters_options_keep_the_llm_order() -> None:
    async def run() -> str:
        async with open_ranker(ScorerConfig()) as ranker:
            return ranker.name

    assert asyncio.run(run()) == KEEP_ORDER


def test_loads_each_adapter_and_sums_their_scores() -> None:
    loaded: list[ScorerAdapter] = []
    tables = {
        ScorerAdapter.PLL: SyncTableScorer({"Read the book.": 1.0, "Read teh book.": -1.0}),
        ScorerAdapter.COLA: SyncTableScorer({"Read the book.": 0.5}),
    }

    def load(adapter: ScorerAdapter, config: ScorerConfig) -> SentenceScorer:
        loaded.append(adapter)
        return tables[adapter]

    config = ScorerConfig(adapters=(ScorerAdapter.PLL, ScorerAdapter.COLA))

    async def run() -> tuple[str, Suggestion]:
        async with open_ranker(config, load=load) as ranker:
            contract: OptionRanker = ranker
            ranking = await contract.rank([TYPO], [SENTENCE])
            return ranker.name, ranking.kept[0]

    name, ranked = asyncio.run(run())

    assert loaded == [ScorerAdapter.PLL, ScorerAdapter.COLA]
    assert ranked.options[0].rating == Scored(score=1.5, delta=2.5)
    assert name == config.name


def test_name_covers_adapters_models_window_and_filters() -> None:
    base = ScorerConfig(adapters=(ScorerAdapter.PLL,))
    filtered = ScorerConfig(
        adapters=(ScorerAdapter.PLL,),
        policy=ScoringPolicy({ErrorKind.PUNCTUATION: KindPolicy(filter=True, margin=0.5)}),
    )

    assert base.name == "pll=distilroberta-base@3|filter="
    assert ScorerConfig(adapters=(ScorerAdapter.PLL,), window=5).name != base.name
    assert ScorerConfig(adapters=(ScorerAdapter.COLA,)).name != base.name
    assert filtered.name == "pll=distilroberta-base@3|filter=Punctuation:0.5"


@pytest.mark.parametrize(
    ("adapter", "module"),
    [(ScorerAdapter.PLL, "langcheck.scoring.pll"), (ScorerAdapter.COLA, "langcheck.scoring.cola")],
)
def test_missing_pytorch_names_the_fix(
    monkeypatch: pytest.MonkeyPatch, adapter: ScorerAdapter, module: str
) -> None:
    # A None entry makes the import fail as it would without the PyTorch extras.
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ScorerUnavailableError, match="just setup cpu"):
        load_adapter(adapter, ScorerConfig())
