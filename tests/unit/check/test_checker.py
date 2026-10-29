import asyncio
import json
import re
from collections.abc import Collection, Sequence
from dataclasses import replace

from langcheck.check.checker import Checker, CheckResult, Unchecked
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.protocols import OptionRanker
from langcheck.check.resolve import RejectionReason, ValidationPolicy
from langcheck.check.suggestion import Option, Scored
from langcheck.context.models import DocumentContext
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError
from langcheck.scoring.policy import KindPolicy, ScoringPolicy
from langcheck.scoring.ranker import KeepOrder, ScoringRanker
from langcheck.taxonomy.kinds import ErrorKind
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter
from tests.conftest import ScriptedLLM, TableScorer

FIXES = {"teh": "the", "recieve": "receive", "Bilbo": "Bill"}
_FOCUS = re.compile(r"^\[(S\d+)\] (.*)$", re.MULTILINE)


def proofreader(messages: Sequence[ChatMessage]) -> str:
    """Replies like a model that knows a few misspellings, for whatever sentences it gets."""
    issues = [
        {
            "sentence": sentence_id,
            "original": wrong,
            "kind": "Spelling",
            "explanation": f"{wrong} is misspelt.",
            "options": [right],
        }
        for sentence_id, sentence in _FOCUS.findall(messages[-1].content)
        for wrong, right in FIXES.items()
        if re.search(rf"\b{wrong}\b", sentence)
    ]
    return json.dumps({"issues": issues})


def make_checker(
    llm: ScriptedLLM,
    *,
    model: str = "m",
    max_words: int = 250,
    ranker: OptionRanker | None = None,
) -> Checker:
    return Checker(
        llm,
        splitter=RegexSentenceSplitter(),
        cache=MemoryResultCache(),
        model=model,
        options=ChatOptions(),
        policy=ValidationPolicy(),
        ranker=ranker or KeepOrder(),
        max_words=max_words,
    )


def check(
    checker: Checker,
    text: str,
    *,
    user: str = "ana",
    context: DocumentContext | None = None,
    dictionary: Collection[str] = (),
) -> CheckResult:
    return asyncio.run(
        checker.check(
            text,
            user=user,
            context=context or DocumentContext.empty(),
            dictionary=dictionary,
        )
    )


def test_finds_and_places_suggestions_in_document_order() -> None:
    text = "I recieve mail.\n\nRead teh book."
    llm = ScriptedLLM([proofreader])

    result = check(make_checker(llm, max_words=3), text)

    assert [(s.span, s.original, s.options) for s in result.suggestions] == [
        (Span(2, 9), "recieve", (Option("receive"),)),
        (Span(22, 25), "teh", (Option("the"),)),
    ]
    assert result.unchecked == ()
    assert len(llm.chats) == 2


def test_unchanged_sentences_come_from_the_cache_at_their_new_offsets() -> None:
    llm = ScriptedLLM([proofreader])
    checker = make_checker(llm)
    first = check(checker, "A one. B two. Read teh book.")

    moved = check(checker, "Changed one. B two. Read teh book.")

    assert len(llm.chats) == 2
    focus = llm.chats[1][-1].content.split("Sentences to check:")[1]
    assert "[S1] Changed one." in focus
    assert "[S2] B two." in focus
    assert "[S3]" not in focus
    [suggestion] = moved.suggestions
    assert suggestion.span == Span(25, 28)
    assert suggestion.id != first.suggestions[0].id


def test_the_cache_is_per_user_and_per_model() -> None:
    llm = ScriptedLLM([proofreader])
    checker = make_checker(llm)

    check(checker, "Read teh book.", user="ana")
    check(checker, "Read teh book.", user="ana")
    check(checker, "Read teh book.", user="ben")

    assert len(llm.chats) == 2


def test_dictionary_and_glossary_words_are_never_flagged() -> None:
    llm = ScriptedLLM([proofreader])
    context = replace(DocumentContext.empty(), glossary=("Bilbo",))

    result = check(make_checker(llm), "Bilbo read teh book.", context=context, dictionary={"teh"})

    assert result.suggestions == ()
    assert {r.reason for r in result.rejections} == {RejectionReason.INTENTIONAL}
    assert [s.original for s in result.filtered] == ["teh"]
    assert "Glossary (never flag): Bilbo\n" in llm.chats[0][-1].content


def test_changing_the_dictionary_reuses_the_cache() -> None:
    llm = ScriptedLLM([proofreader])
    checker = make_checker(llm)

    plain = check(checker, "Read teh book.")
    with_word = check(checker, "Read teh book.", dictionary={"teh"})
    without_word = check(checker, "Read teh book.", dictionary=set())

    assert len(llm.chats) == 1
    assert [s.original for s in plain.suggestions] == ["teh"]
    assert with_word.suggestions == ()
    assert without_word.suggestions == plain.suggestions


def test_failed_chunks_are_reported_unchecked_and_not_cached() -> None:
    llm = ScriptedLLM([LLMError("down"), proofreader])
    checker = make_checker(llm)

    failed = check(checker, "Read teh book.")
    retried = check(checker, "Read teh book.")

    assert failed.suggestions == ()
    assert failed.unchecked == (Unchecked(Span(0, 14), "LLM error: down"),)
    assert [s.original for s in retried.suggestions] == ["teh"]


def test_empty_text_needs_no_llm_call() -> None:
    llm = ScriptedLLM([proofreader])

    assert check(make_checker(llm), "  \n") == CheckResult((), (), ())
    assert llm.chats == []


def test_ranked_options_are_cached_and_filtered_suggestions_reported() -> None:
    llm = ScriptedLLM([proofreader])
    scorer = TableScorer({"Read the book.": 1.0, "I receive mail.": -1.0})
    policy = ScoringPolicy({ErrorKind.SPELLING: KindPolicy(filter=True)})
    checker = make_checker(llm, ranker=ScoringRanker(scorer, policy=policy, name="table"))

    first = check(checker, "Read teh book. I recieve mail.")
    again = check(checker, "Read teh book. I recieve mail.")

    [kept] = first.suggestions
    assert kept.original == "teh"
    assert kept.options[0].rating == Scored(score=1.0, delta=1.0)
    assert [s.original for s in first.filtered] == ["recieve"]
    assert again.suggestions == first.suggestions
    assert len(llm.chats) == 1
    assert len(scorer.batches) == 1
