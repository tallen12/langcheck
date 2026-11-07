import asyncio
import json

from langcheck.judging.llm_judge import (
    JUDGE_PROMPT_VERSION,
    LLMEditJudge,
    describe_edit,
    judge_messages,
    judge_name,
)
from langcheck.judging.models import Candidate, Item, Judged, NotJudged, Verdict
from langcheck.llm.models import LLMError, Role
from tests.conftest import ScriptedLLM

TOKENS = ("He", "go", "to", "school", "yesterday", ".")
ITEM = Item(
    TOKENS,
    "It rained.",
    (
        Candidate(1, 2, ("went",), frozenset()),
        Candidate(3, 3, ("the",), frozenset()),
    ),
)


def reply(*verdicts: tuple[int, str], left: bool = False) -> str:
    edits = [{"edit": n, "reason": "because", "verdict": v} for n, v in verdicts]
    return json.dumps({"edits": edits, "remaining_reason": "ok", "remaining_error": left})


def test_the_prompt_shows_context_the_sentence_and_each_edit_applied_alone() -> None:
    system, user = judge_messages(ITEM)

    assert system.role is Role.SYSTEM
    assert '"neutral"' in system.content
    assert user.content.splitlines() == [
        "Preceding text: It rained.",
        "",
        "Original: He go to school yesterday.",
        "",
        "Edits:",
        '1. replace "go" with "went"',
        "   Result: He went to school yesterday.",
        '2. insert "the" after "to"',
        "   Result: He go to the school yesterday.",
    ]


def test_edits_are_described_as_replacements_insertions_and_deletions() -> None:
    assert (
        describe_edit(TOKENS, Candidate(0, 0, ("So",), frozenset())) == 'insert "So" at the start'
    )
    assert describe_edit(TOKENS, Candidate(4, 5, (), frozenset())) == 'delete "yesterday"'


def test_a_sentence_without_edits_asks_only_whether_an_error_is_left() -> None:
    _, user = judge_messages(Item(TOKENS, "", ()))

    assert user.content.startswith("Original:")
    assert "Edits: none" in user.content


def test_verdicts_come_back_in_candidate_order_without_reasoning() -> None:
    llm = ScriptedLLM([reply((2, "neutral"), (1, "valid"), left=True)])
    judge = LLMEditJudge(llm, "qwen3:14b")

    result = asyncio.run(judge.judge(ITEM))

    assert result == Judged(((Verdict.VALID, "because"), (Verdict.NEUTRAL, "because")), True, "ok")
    assert not llm.options[0].think
    assert judge.name == f"llm:qwen3:14b@{JUDGE_PROMPT_VERSION}"
    assert judge.name == judge_name("qwen3:14b", think=False)


def test_judges_that_reason_are_stored_apart() -> None:
    llm = ScriptedLLM([reply((1, "valid"), (2, "valid"))])
    judge = LLMEditJudge(llm, "qwen3:14b", think=True)

    asyncio.run(judge.judge(ITEM))

    assert llm.options[0].think
    assert judge.name == f"llm:qwen3:14b@{JUDGE_PROMPT_VERSION}+think"


def test_a_reply_that_misnumbers_the_edits_is_not_judged() -> None:
    judge = LLMEditJudge(ScriptedLLM([reply((1, "valid"), (1, "wrong"))]), "m", think=False)

    result = asyncio.run(judge.judge(ITEM))

    assert isinstance(result, NotJudged)
    assert "expected verdicts on edits 1 to 2" in result.reason


def test_a_server_failure_is_not_judged() -> None:
    judge = LLMEditJudge(ScriptedLLM([LLMError("down")]), "m")

    assert asyncio.run(judge.judge(ITEM)) == NotJudged("LLM error: down")
