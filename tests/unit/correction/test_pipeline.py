import asyncio
import json
from collections.abc import Callable, Sequence

from langcheck.correction.llm_corrector import LlmCorrector
from langcheck.correction.models import (
    Corrected,
    CorrectionPolicy,
    FlagOutcome,
    RejectionReason,
    Unavailable,
)
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError
from langcheck.tagging.reading import PairedReader
from langcheck.tagging.tiers import TierPolicy
from tests.conftest import Responder, ScriptedLLM, WordTableTagger

type Script = Sequence[str | LLMError | Responder]


def reply(*corrections: tuple[str, str], flag: str = "F1") -> str:
    return json.dumps(
        {
            "flags": [
                {
                    "flag": flag,
                    "corrections": [
                        {"original": o, "replacement": r, "kind": "VerbForm", "explanation": "Why"}
                        for o, r in corrections
                    ],
                }
            ]
        }
    )


def suggest(
    text: str,
    table: dict[str, float],
    script: Script,
    build: Callable[[Script], ScriptedLLM],
) -> tuple[list[FlagOutcome], ScriptedLLM]:
    llm = build(script)
    pipeline = CorrectionPipeline(
        corrector=LlmCorrector(llm, CorrectionPolicy(count=3), ChatOptions()),
        reader=PairedReader(WordTableTagger(table)),
        tiers=TierPolicy(),
        policy=CorrectionPolicy(count=3),
        context_sentences=1,
    )
    return asyncio.run(pipeline.suggest(text)), llm


def test_corrections_come_back_ranked_with_offsets_into_the_text(
    scripted_llm: Callable[[Script], ScriptedLLM],
) -> None:
    table = {"go": 0.9, "gone": 0.4}

    [outcome], _ = suggest(
        "Fine here. He go home.", table, [reply(("", "gone"), ("", "goes"))], scripted_llm
    )

    assert isinstance(outcome, Corrected)
    assert [r.candidate.replacement for r in outcome.ranked] == ["goes", "gone"]
    assert outcome.ranked[0].candidate.span.start == 14


def test_the_llm_is_asked_about_flagged_sentences_only_with_their_context(
    scripted_llm: Callable[[Script], ScriptedLLM],
) -> None:
    _, llm = suggest("Fine here. He go home.", {"go": 0.9}, [reply(("", "goes"))], scripted_llm)

    [chat] = llm.chats
    last: ChatMessage = chat[-1]
    request = json.loads(last.content)
    assert request["sentence"] == "He go home."
    assert request["context"] == "Fine here."
    assert request["flags"][0]["start"] == 3


def test_clean_text_never_reaches_the_llm(
    scripted_llm: Callable[[Script], ScriptedLLM],
) -> None:
    outcomes, llm = suggest("All fine here.", {}, [reply()], scripted_llm)

    assert outcomes == []
    assert llm.chats == []


def test_a_flag_the_llm_finds_correct_has_no_corrections(
    scripted_llm: Callable[[Script], ScriptedLLM],
) -> None:
    [outcome], _ = suggest("He go home.", {"go": 0.9}, [reply()], scripted_llm)

    assert isinstance(outcome, Corrected)
    assert (outcome.ranked, outcome.rejected) == ((), ())


def test_an_llm_failure_leaves_each_flag_of_the_sentence_unavailable(
    scripted_llm: Callable[[Script], ScriptedLLM],
) -> None:
    outcomes, _ = suggest(
        "He go to teh shop.", {"go": 0.9, "teh": 0.8}, [LLMError("down")], scripted_llm
    )

    assert [type(outcome) for outcome in outcomes] == [Unavailable, Unavailable]


def test_placing_and_judging_rejections_both_reach_the_outcome(
    scripted_llm: Callable[[Script], ScriptedLLM],
) -> None:
    [outcome], _ = suggest(
        "He go home.",
        {"go": 0.9},
        [reply(("home", "goes"), ("", "go!"), ("", "goes"))],
        scripted_llm,
    )

    assert isinstance(outcome, Corrected)
    assert [r.candidate.replacement for r in outcome.ranked] == ["goes"]
    assert [r.reason for r in outcome.rejected] == [
        RejectionReason.NOT_AT_FLAG,
        RejectionReason.STILL_FLAGGED,
    ]
