"""``EditJudge`` on an OpenAI-compatible LLM: one request per sentence, all its edits at once.

Each annotator's or run's correction is shown whole, with its edits listed under it, and
every edit is judged as part of that correction: an edit that only works with another,
such as half of a moved word, is said to be linked to it rather than judged wrong alone.
When several corrections compete, the judge also says which fixes the sentence best.
Judging a sentence in one request costs one call instead of one per edit.
"""

import string
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from langcheck.judging.models import (
    Candidate,
    EditVerdict,
    Item,
    Judged,
    JudgeResult,
    NoneBetter,
    NotJudged,
    Preference,
    Preferred,
    Tied,
    Verdict,
)
from langcheck.judging.render import render, written_span
from langcheck.llm.models import ChatMessage, ChatOptions, Role
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.llm.structured import Failed, Parsed, complete_structured

JUDGE_PROMPT_VERSION = "judge-v2"
LLM_JUDGE_PREFIX = "llm"
THINK_SUFFIX = "+think"
TIE = "tie"
NONE = "none"
LETTERS = string.ascii_uppercase
REASON_CHARACTERS = 200
"""Longest reason kept. The reply schema says so, and servers that constrain replies to
the schema stop a model that repeats itself inside a reason, which once ran until the
proxy timed out."""
MAX_TOKENS = 1024
"""Tokens a judge may write without reasoning; a reply in the schema needs a few hundred."""
THINKING_TOKENS = 8192
"""Tokens with reasoning on, which come before the reply and count toward the limit."""

SYSTEM_PROMPT = """You judge corrections of English sentences written by learners and \
by fluent writers. Each correction (A, B, ...) is one person's whole rewrite of the \
sentence, with its edits numbered below it; an edit shared by two corrections has one \
number. Judge each edit as part of its correction, with that correction's other edits \
applied:

- "valid": the original has a real error (grammar, spelling, punctuation, word choice, \
agreement, a missing or extra word) and the edit fixes it without changing the meaning.
- "neutral": the original was already acceptable; the edit only changes style, wording \
or formality, or is an optional comma.
- "wrong": the edit introduces an error, changes the meaning, or fixes the error badly.

For each edit, list in "linked" the numbers of other edits in the same correction that \
it only works together with: the removal and insertion of a moved word, a pair of quote \
marks or brackets, or words that must agree. Leave it empty when the edit stands alone.

When there are several corrections, set "preferred" to the letter of the one that fixes \
the sentence best, "tie" if the best are equally good, or "none" if none improves it. \
Then say whether the sentence, with only the valid edits of the best correction applied, \
still contains a real error. Ignore style. Give each reason in one short sentence. Reply \
with JSON only."""


def _clipped(value: object) -> object:
    return value[:REASON_CHARACTERS] if isinstance(value, str) else value


type Reason = Annotated[str, BeforeValidator(_clipped), Field(max_length=REASON_CHARACTERS)]
"""One short sentence; a longer one from a server that ignores the schema is cut."""


class _EditReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    edit: int = Field(ge=1)
    reason: Reason
    verdict: Verdict
    linked: tuple[int, ...] = ()


class JudgeReply(BaseModel):
    """A judge's reply: a verdict per edit, the best correction and whether an error
    remains.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    edits: tuple[_EditReply, ...]
    preferred_reason: Reason
    preferred: str
    remaining_reason: Reason
    remaining_error: bool


class LLMEditJudge:
    """Asks one model, by name, to judge every edit of a sentence in one request."""

    def __init__(self, llm: AsyncLLMClient, model: str, *, think: bool = False) -> None:
        self._llm = llm
        self._model = model
        self._think = think
        self._options = ChatOptions(
            think=think,
            schema_name="judgment",
            max_tokens=THINKING_TOKENS if think else MAX_TOKENS,
        )

    @property
    def name(self) -> str:
        """``llm:<model>@<prompt version>``, plus ``+think`` with reasoning on: a new prompt
        version or reasoning mode judges again.
        """
        return judge_name(self._model, think=self._think)

    async def judge(self, item: Item) -> JudgeResult:
        """A verdict on each candidate of ``item``; ``NotJudged`` if the reply is unusable."""
        result = await complete_structured(
            self._llm, judge_messages(item), JudgeReply, options=self._options
        )
        match result:
            case Failed(reason):
                return NotJudged(reason)
            case Parsed(reply):
                return judged(reply, item)


def judge_name(model: str, *, think: bool) -> str:
    """The store name of the LLM judge on ``model`` with the current prompt."""
    suffix = THINK_SUFFIX if think else ""
    return f"{LLM_JUDGE_PREFIX}:{model}@{JUDGE_PROMPT_VERSION}{suffix}"


def judge_messages(item: Item) -> list[ChatMessage]:
    """The chat asking for verdicts on ``item``'s edits, correction by correction."""
    numbers = {candidate.key: n for n, candidate in enumerate(item.candidates, start=1)}
    by_key = {candidate.key: candidate for candidate in item.candidates}
    lines: list[str] = []
    if item.context:
        lines += [f"Preceding text: {item.context}", ""]
    lines += [f"Original: {render(item.tokens, [], item.written)}", ""]
    if not item.versions:
        lines.append('Corrections: none. Reply with an empty edits list and "none" preferred.')
    for letter, version in zip(LETTERS, item.versions, strict=False):
        lines.append(
            f"{letter} ({version.label}): {render(item.tokens, version.edits, item.written)}"
        )
        for key in version.edits:
            lines.append(f"   {numbers[key]}. {describe_edit(item, by_key[key])}")
    return [ChatMessage(Role.SYSTEM, SYSTEM_PROMPT), ChatMessage(Role.USER, "\n".join(lines))]


def describe_edit(item: Item, candidate: Candidate) -> str:
    """The edit as a reader sees it: what it replaces, inserts or deletes."""
    before = written_span(item.tokens, candidate.start, candidate.end, item.written)
    after = " ".join(candidate.correction)
    if not before:
        anchor = written_span(item.tokens, candidate.start - 1, candidate.start, item.written)
        where = f'after "{anchor}"' if candidate.start > 0 else "at the start"
        return f'insert "{after}" {where}'
    if not after:
        return f'delete "{before}"'
    return f'replace "{before}" with "{after}"'


def judged(reply: JudgeReply, item: Item) -> JudgeResult:
    """``reply`` as verdicts in candidate order; ``NotJudged`` unless it rates each edit
    exactly once, since a verdict given to the wrong number would be stored on the wrong
    edit, and names a correction that exists. Links to edits outside the edit's own
    corrections are dropped.
    """
    count = len(item.candidates)
    by_number = {verdict.edit: verdict for verdict in reply.edits}
    if len(by_number) != len(reply.edits) or set(by_number) != set(range(1, count + 1)):
        numbers = sorted(verdict.edit for verdict in reply.edits)
        return NotJudged(f"expected verdicts on edits 1 to {count}, got {numbers}")
    preference = _preference(reply, len(item.versions))
    if isinstance(preference, NotJudged):
        return preference
    companions = _companions(item)
    verdicts = tuple(
        EditVerdict(
            by_number[n].verdict,
            by_number[n].reason,
            tuple(sorted({m - 1 for m in by_number[n].linked if m - 1 in companions[n - 1]})),
        )
        for n in range(1, count + 1)
    )
    return Judged(verdicts, reply.remaining_error, reply.remaining_reason, preference)


def _preference(reply: JudgeReply, versions: int) -> Preference | NotJudged | None:
    if versions < 2:
        return None
    answer = reply.preferred.strip().strip(".").lower()
    if answer == TIE:
        return Tied(reply.preferred_reason)
    if answer == NONE:
        return NoneBetter(reply.preferred_reason)
    letters: list[str] = [letter.lower() for letter in LETTERS[:versions]]
    if answer in letters:
        return Preferred(letters.index(answer), reply.preferred_reason)
    return NotJudged(f"preferred {reply.preferred!r} is not one of {LETTERS[:versions]}, tie, none")


def _companions(item: Item) -> list[set[int]]:
    """For each candidate, the indices of the other candidates it shares a correction with."""
    index = {candidate.key: i for i, candidate in enumerate(item.candidates)}
    found: list[set[int]] = [set() for _ in item.candidates]
    for version in item.versions:
        members = {index[key] for key in version.edits}
        for member in members:
            found[member] |= members - {member}
    return found
