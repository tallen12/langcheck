import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from langcheck.check.checker import CheckResult
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.suggestion import Confidence, Scored
from langcheck.context.models import DocumentContext
from langcheck.correction.checker import CorrectionChecker, open_correction_checker
from langcheck.correction.models import CorrectionPolicy
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.llm.models import ChatOptions, LLMError
from langcheck.tagging.loaded import Trained
from langcheck.tagging.models import SlotScores
from langcheck.tagging.protocols import SlotScorer
from langcheck.tagging.tiers import TierPolicy
from langcheck.taxonomy.kinds import ErrorKind
from tests.conftest import Responder, ScriptedLLM, WordTableTagger

type Script = Sequence[str | LLMError | Responder]
type Build = Callable[[Script], ScriptedLLM]
EMPTY = DocumentContext.empty()


def reply(*corrections: tuple[str, str]) -> str:
    return json.dumps(
        {
            "flags": [
                {
                    "flag": "F1",
                    "corrections": [
                        {"original": o, "replacement": r, "kind": "VerbForm", "explanation": "Why"}
                        for o, r in corrections
                    ],
                }
            ]
        }
    )


class MissingWordTagger:
    """Finds one gap wrong until a sentence holds the missing word."""

    def __init__(self, *, gap: int = 2) -> None:
        self._gap = gap

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        return [
            SlotScores(
                (0.0,) * len(s),
                tuple(0.9 if i == self._gap and "the" not in s else 0.0 for i in range(len(s) + 1)),
            )
            for s in sentences
        ]


def checker(
    llm: ScriptedLLM, table: dict[str, float], *, scorer: SlotScorer | None = None
) -> CorrectionChecker:
    pipeline = CorrectionPipeline(
        llm=llm,
        scorer=scorer if scorer is not None else WordTableTagger(table),
        commas=None,
        tiers=TierPolicy(),
        policy=CorrectionPolicy(),
        options=ChatOptions(),
        context_sentences=0,
    )
    return CorrectionChecker(pipeline, cache=MemoryResultCache(), model="m", name="t")


def check(subject: CorrectionChecker, text: str, *, dictionary: Sequence[str] = ()) -> CheckResult:
    return asyncio.run(subject.check(text, user="u", context=EMPTY, dictionary=dictionary))


def test_a_flag_becomes_one_suggestion_with_ranked_options(scripted_llm: Build) -> None:
    subject = checker(scripted_llm([reply(("", "gone"), ("", "goes"))]), {"go": 0.9, "gone": 0.5})

    [suggestion] = check(subject, "Fine. He go home.").suggestions

    assert suggestion.span.start == 9
    assert suggestion.original == "go"
    assert suggestion.kind is ErrorKind.VERB_FORM
    assert [o.replacement for o in suggestion.options] == ["goes", "gone"]
    assert [o.confidence for o in suggestion.options] == [Confidence.HIGH, Confidence.LOW]
    assert suggestion.confidence is Confidence.HIGH
    assert suggestion.options[0].rating == Scored(-0.0, 0.9)


def test_a_weak_best_option_makes_the_suggestion_low_confidence(scripted_llm: Build) -> None:
    subject = checker(scripted_llm([reply(("", "gone"))]), {"go": 0.9, "gone": 0.5})

    [suggestion] = check(subject, "He go home.").suggestions

    assert suggestion.confidence is Confidence.LOW


def test_corrections_of_different_spans_share_the_span_covering_them(scripted_llm: Build) -> None:
    subject = checker(scripted_llm([reply(("", "goes"), ("go home", "went home"))]), {"go": 0.9})

    [suggestion] = check(subject, "He go home.").suggestions

    assert suggestion.original == "go home"
    assert [o.replacement for o in suggestion.options] == ["goes home", "went home"]


def test_a_flag_with_nothing_to_offer_is_not_reported(scripted_llm: Build) -> None:
    subject = checker(scripted_llm([reply()]), {"go": 0.9})

    result = check(subject, "He go home.")

    assert result.suggestions == ()
    assert result.unchecked == ()


def test_an_llm_failure_leaves_the_sentence_unchecked(scripted_llm: Build) -> None:
    subject = checker(scripted_llm([LLMError("down")]), {"go": 0.9})

    result = check(subject, "Fine. He go home.")

    [unchecked] = result.unchecked
    assert (unchecked.span.start, unchecked.span.end) == (6, 17)
    assert "down" in unchecked.reason


def test_a_failed_sentence_is_tried_again(scripted_llm: Build) -> None:
    llm = scripted_llm([LLMError("down"), reply(("", "goes"))])
    subject = checker(llm, {"go": 0.9})

    check(subject, "He go home.")

    assert len(check(subject, "He go home.").suggestions) == 1


def test_an_unchanged_sentence_is_not_sent_again_even_when_it_moved(scripted_llm: Build) -> None:
    llm = scripted_llm([reply(("", "goes"))])
    subject = checker(llm, {"go": 0.9})

    check(subject, "He go home.")
    [suggestion] = check(subject, "All fine. He go home.").suggestions

    assert len(llm.chats) == 1
    assert suggestion.span.start == 13


def test_a_glossary_listing_an_error_does_not_hide_its_correction(scripted_llm: Build) -> None:
    subject = checker(scripted_llm([reply(("", "goes"))]), {"go": 0.9})
    context = replace(EMPTY, glossary=("go",))

    result = asyncio.run(subject.check("He go home.", user="u", context=context))

    assert len(result.suggestions) == 1


def test_dictionary_words_are_dropped_after_the_cache(scripted_llm: Build) -> None:
    llm = scripted_llm([reply(("", "goes"))])
    subject = checker(llm, {"go": 0.9})

    result = check(subject, "He go home.", dictionary=["go"])
    again = check(subject, "He go home.")

    assert result.suggestions == ()
    assert len(result.filtered) == 1
    assert len(again.suggestions) == 1
    assert len(llm.chats) == 1


def test_an_insertion_takes_in_the_word_before_it(scripted_llm: Build) -> None:
    gap = checker(scripted_llm([reply(("", "the"))]), {}, scorer=MissingWordTagger())

    [suggestion] = check(gap, "Near store today.").suggestions

    assert suggestion.original == "store"
    assert suggestion.options[0].replacement == "store the"


def test_an_insertion_at_the_start_takes_in_the_word_after_it(scripted_llm: Build) -> None:
    gap = checker(scripted_llm([reply(("", "the"))]), {}, scorer=MissingWordTagger(gap=0))

    [suggestion] = check(gap, "Store today.").suggestions

    assert suggestion.original == "Store"
    assert suggestion.options[0].replacement == "the Store"


def test_open_loads_the_tagger_with_the_configured_device(
    scripted_llm: Build, tmp_path: Path
) -> None:
    seen: list[tuple[Path, str]] = []

    def load(path: Path, *, device: str) -> Trained:
        seen.append((path, device))
        return Trained(WordTableTagger({}), 0.5, "base", 0.5, None, 2)

    class Config:
        device = "cpu"

        def policy(self) -> CorrectionPolicy:
            return CorrectionPolicy()

    subject = asyncio.run(
        open_correction_checker(
            scripted_llm([reply()]), Config(), tmp_path / "tagger", model="m", load=load
        )
    )

    assert seen == [(tmp_path / "tagger", "cpu")]
    assert check(subject, "All fine.").suggestions == ()
