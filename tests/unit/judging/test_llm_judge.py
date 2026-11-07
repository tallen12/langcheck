import asyncio
import json

from langcheck.evaluation.models import ScoredEdit
from langcheck.judging.llm_judge import (
    JUDGE_PROMPT_VERSION,
    LLMEditJudge,
    describe_edit,
    judge_messages,
    judge_name,
)
from langcheck.judging.models import (
    EditVerdict,
    Item,
    Judged,
    NoneBetter,
    NotJudged,
    Preferred,
    Tied,
    Verdict,
)
from langcheck.judging.panel import pool_item
from langcheck.llm.models import LLMError, Role
from tests.conftest import ScriptedLLM

TOKENS = ("A", "build", "-", "up", "often", "go", "bad", ".")
WRITTEN = "A build-up often go bad."
GOES = ScoredEdit(5, 6, ("goes",), None)
WENT = ScoredEdit(5, 6, ("went",), None)
MOVE_OUT = ScoredEdit(4, 5, (), None)
MOVE_IN = ScoredEdit(7, 7, ("often",), None)
ITEM = pool_item(
    TOKENS,
    gold=[(GOES, MOVE_OUT, MOVE_IN), (WENT,)],
    context="It rained.",
    written=WRITTEN,
)


def reply(
    *verdicts: tuple[int, str] | tuple[int, str, list[int]],
    preferred: str = "A",
    left: bool = False,
) -> str:
    edits = [
        {"edit": v[0], "reason": "because", "verdict": v[1], "linked": v[2] if len(v) > 2 else []}
        for v in verdicts
    ]
    return json.dumps(
        {
            "edits": edits,
            "preferred_reason": "better",
            "preferred": preferred,
            "remaining_reason": "ok",
            "remaining_error": left,
        }
    )


def test_each_correction_is_shown_whole_in_the_published_spelling_with_its_edits() -> None:
    system, user = judge_messages(ITEM)

    assert system.role is Role.SYSTEM
    assert '"linked"' in system.content
    assert user.content.splitlines() == [
        "Preceding text: It rained.",
        "",
        "Original: A build-up often go bad.",
        "",
        "A (annotator 1): A build-up goes bad often.",
        '   1. delete "often"',
        '   2. replace "go" with "goes"',
        '   4. insert "often" after "bad"',
        "B (annotator 2): A build-up often went bad.",
        '   3. replace "go" with "went"',
    ]


def test_edits_are_described_as_replacements_insertions_and_deletions() -> None:
    item = pool_item(TOKENS, gold=[(ScoredEdit(0, 0, ("So",), None),)])

    assert describe_edit(item, item.candidates[0]) == 'insert "So" at the start'


def test_a_sentence_without_edits_asks_only_whether_an_error_is_left() -> None:
    _, user = judge_messages(Item(TOKENS, "", ()))

    assert user.content.startswith("Original: A build - up often go bad.")
    assert "Corrections: none" in user.content


def test_verdicts_links_and_the_preferred_correction_come_back_in_order() -> None:
    llm = ScriptedLLM(
        [reply((4, "valid", [1, 3]), (2, "valid"), (3, "valid"), (1, "valid", [4]), preferred="b.")]
    )
    judge = LLMEditJudge(llm, "qwen3:14b")

    result = asyncio.run(judge.judge(ITEM))

    assert result == Judged(
        (
            EditVerdict(Verdict.VALID, "because", (3,)),
            EditVerdict(Verdict.VALID, "because"),
            EditVerdict(Verdict.VALID, "because"),
            # Edit 3 belongs to another correction, so that link is dropped.
            EditVerdict(Verdict.VALID, "because", (0,)),
        ),
        False,
        "ok",
        Preferred(1, "better"),
    )
    assert not llm.options[0].think
    assert judge.name == f"llm:qwen3:14b@{JUDGE_PROMPT_VERSION}"
    assert judge.name == judge_name("qwen3:14b", think=False)


def test_ties_and_no_improvement_are_preferences_too() -> None:
    verdicts = [(n, "neutral") for n in range(1, 5)]

    tied = asyncio.run(
        LLMEditJudge(ScriptedLLM([reply(*verdicts, preferred="Tie")]), "m").judge(ITEM)
    )
    none = asyncio.run(
        LLMEditJudge(ScriptedLLM([reply(*verdicts, preferred="none")]), "m").judge(ITEM)
    )

    assert isinstance(tied, Judged) and tied.preference == Tied("better")
    assert isinstance(none, Judged) and none.preference == NoneBetter("better")


def test_a_single_correction_has_no_preference() -> None:
    item = pool_item(TOKENS, gold=[(GOES,)])
    judge = LLMEditJudge(ScriptedLLM([reply((1, "valid"), preferred="anything")]), "m")

    result = asyncio.run(judge.judge(item))

    assert isinstance(result, Judged) and result.preference is None


def test_judges_that_reason_are_stored_apart() -> None:
    llm = ScriptedLLM([reply(*[(n, "valid") for n in range(1, 5)])])
    judge = LLMEditJudge(llm, "qwen3:14b", think=True)

    asyncio.run(judge.judge(ITEM))

    assert llm.options[0].think
    assert judge.name == f"llm:qwen3:14b@{JUDGE_PROMPT_VERSION}+think"


def test_a_reply_that_misnumbers_edits_or_names_no_correction_is_not_judged() -> None:
    misnumbered = LLMEditJudge(ScriptedLLM([reply((1, "valid"), (1, "wrong"))]), "m")
    unknown = LLMEditJudge(
        ScriptedLLM([reply(*[(n, "valid") for n in range(1, 5)], preferred="C")]), "m"
    )

    first = asyncio.run(misnumbered.judge(ITEM))
    second = asyncio.run(unknown.judge(ITEM))

    assert isinstance(first, NotJudged)
    assert "expected verdicts on edits 1 to 4" in first.reason
    assert isinstance(second, NotJudged)
    assert "preferred 'C' is not one of AB" in second.reason


def test_a_server_failure_is_not_judged() -> None:
    judge = LLMEditJudge(ScriptedLLM([LLMError("down")]), "m")

    assert asyncio.run(judge.judge(ITEM)) == NotJudged("LLM error: down")
