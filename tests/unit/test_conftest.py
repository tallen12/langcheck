import asyncio
from collections.abc import Callable, Sequence

import pytest
from transformers import PreTrainedTokenizerFast

from langcheck.auth.models import User, UserExistsError
from langcheck.auth.protocols import ApiKeyStore
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError, Role
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import AsyncSentenceScorer, SentenceScorer
from langcheck.text.models import Span
from tests.conftest import (
    MemoryApiKeyStore,
    Responder,
    ScriptedLLM,
    SyncTableScorer,
    TableScorer,
    WordTableTagger,
)

HI = [ChatMessage(Role.USER, "hi")]


def test_scripted_llm_replies_in_order_repeats_the_last_and_records_chats(
    scripted_llm: Callable[[Sequence[str | LLMError | Responder]], ScriptedLLM],
) -> None:
    llm = scripted_llm(["one", lambda chat: chat[-1].content.upper()])
    client: AsyncLLMClient = llm

    async def run() -> list[str]:
        return [await client.complete(HI, options=ChatOptions()) for _ in range(3)]

    assert asyncio.run(run()) == ["one", "HI", "HI"]
    assert llm.chats == [HI, HI, HI]
    assert len(llm.options) == 3


def test_scripted_llm_raises_scripted_errors() -> None:
    llm = ScriptedLLM([LLMError("down")])

    with pytest.raises(LLMError, match="down"):
        asyncio.run(llm.complete(HI, options=ChatOptions()))


def test_scripted_llm_needs_a_script() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        ScriptedLLM([])


def test_memory_api_key_store_keeps_one_key_per_user() -> None:
    store: ApiKeyStore = MemoryApiKeyStore()

    store.add("ana", "h1")
    with pytest.raises(UserExistsError):
        store.add("ana", "h2")

    assert store.user_for("h1") == User("ana")
    assert store.user_for("h2") is None
    assert store.users() == ["ana"]
    assert store.revoke("ana") is True
    assert store.revoke("ana") is False


def test_table_scorers_look_scores_up_and_record_batches() -> None:
    texts = [ScoredText("a", Span(0, 1)), ScoredText("b", Span(0, 1))]
    asynchronous = TableScorer({"a": 1.5}, default=-1.0)
    blocking = SyncTableScorer({"b": 2.0})
    async_contract: AsyncSentenceScorer = asynchronous
    sync_contract: SentenceScorer = blocking

    assert asyncio.run(async_contract.score(texts)) == [1.5, -1.0]
    assert sync_contract.score(texts) == [0.0, 2.0]
    assert asynchronous.batches == [texts]
    assert blocking.batches == [texts]


def test_tiny_tokenizer_has_roberta_special_tokens_and_offsets(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    encoded = tiny_tokenizer("The dog sleeps.", return_offsets_mapping=True)
    truncated = tiny_tokenizer("The dog sleeps in the garden.", truncation=True, max_length=4)

    assert tiny_tokenizer.mask_token_id is not None
    assert tiny_tokenizer.pad_token_id is not None
    assert encoded["input_ids"][0] == tiny_tokenizer.cls_token_id
    assert encoded["input_ids"][-1] == tiny_tokenizer.sep_token_id
    assert encoded["offset_mapping"][0] == (0, 0)
    assert len(truncated["input_ids"]) == 4


def test_fake_uno_provides_the_modules_the_extension_imports(fake_uno: None) -> None:
    from com.sun.star.lang import Locale
    from com.sun.star.linguistic2 import ProofreadingResult

    locale = Locale("en", "GB", "")
    result = ProofreadingResult()
    result.aText = "Hi."

    assert (locale.Language, locale.Country) == ("en", "GB")
    assert result.aText == "Hi."


def test_token_diff_annotator_diffs_whitespace_tokens() -> None:
    from langcheck.evaluation.m2 import GoldEdit
    from langcheck.evaluation.models import ScoredEdit
    from langcheck.taxonomy.kinds import ErrorKind
    from tests.conftest import TokenDiffAnnotator

    annotator = TokenDiffAnnotator(ErrorKind.SPELLING)

    assert annotator.system_edits(("a", "teh", "b"), "a the b") == [
        ScoredEdit(1, 2, ("the",), ErrorKind.SPELLING)
    ]
    assert annotator.reference_edits(("a",), ("a", "b")) == [
        ScoredEdit(1, 1, ("b",), ErrorKind.SPELLING)
    ]
    assert annotator.gold_edits(("a",), [GoldEdit(0, 1, (), "U:X")])[0].kind is ErrorKind.SPELLING


def test_word_table_tagger_scores_words_from_its_table() -> None:
    tagger = WordTableTagger({"go": 0.9})

    [scores] = tagger.score([("He", "go")])

    assert scores.words == (0.0, 0.9)
    assert scores.gaps == (0.0, 0.0, 0.0)
    assert tagger.batches == [[("He", "go")]]
