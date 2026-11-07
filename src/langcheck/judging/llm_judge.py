"""``EditJudge`` on an OpenAI-compatible LLM: one request per sentence, all its edits at once.

Judging a sentence's edits together costs one request instead of one per edit, and lets
the same answer say whether an error is left once the valid edits are applied. Each edit
is still shown on its own, applied alone, so the judge rates it without the others.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from langcheck.evaluation.detokenize import detokenize
from langcheck.judging.models import Candidate, Item, Judged, JudgeResult, NotJudged, Verdict
from langcheck.llm.models import ChatMessage, ChatOptions, Role
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.llm.structured import Failed, Parsed, complete_structured

JUDGE_PROMPT_VERSION = "judge-v1"
LLM_JUDGE_PREFIX = "llm"
THINK_SUFFIX = "+think"

SYSTEM_PROMPT = """You judge corrections to English sentences written by learners and \
by fluent writers. For each numbered edit, applied alone to the original sentence, decide:

- "valid": the original has a real error (grammar, spelling, punctuation, word choice, \
agreement, a missing or extra word) and the edit fixes it without changing the meaning.
- "neutral": the original was already acceptable; the edit only changes style, wording \
or formality, or is an optional comma.
- "wrong": the edit introduces an error, changes the meaning, or fixes the error badly.

Judge each edit on its own. Several edits may overlap or compete; more than one can be \
valid. Then say whether the sentence, with only the edits you judged valid applied, \
still contains a real error. Ignore style. Give each reason in one short sentence. \
Reply with JSON only."""


class _EditVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    edit: int = Field(ge=1)
    reason: str
    verdict: Verdict


class JudgeReply(BaseModel):
    """A judge's reply: a verdict per edit and whether an error remains."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    edits: tuple[_EditVerdict, ...]
    remaining_reason: str
    remaining_error: bool


class LLMEditJudge:
    """Asks one model, by name, to judge every edit of a sentence in one request."""

    def __init__(self, llm: AsyncLLMClient, model: str, *, think: bool = False) -> None:
        self._llm = llm
        self._model = model
        self._think = think
        self._options = ChatOptions(think=think, schema_name="judgment")

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
                return judged(reply, len(item.candidates))


def judge_name(model: str, *, think: bool) -> str:
    """The store name of the LLM judge on ``model`` with the current prompt."""
    suffix = THINK_SUFFIX if think else ""
    return f"{LLM_JUDGE_PREFIX}:{model}@{JUDGE_PROMPT_VERSION}{suffix}"


def judge_messages(item: Item) -> list[ChatMessage]:
    """The chat asking for verdicts on ``item``'s edits."""
    lines: list[str] = []
    if item.context:
        lines += [f"Preceding text: {item.context}", ""]
    lines += [f"Original: {detokenize(item.tokens)}", ""]
    if item.candidates:
        lines.append("Edits:")
        for number, candidate in enumerate(item.candidates, start=1):
            lines.append(f"{number}. {describe_edit(item.tokens, candidate)}")
            lines.append(f"   Result: {detokenize(applied(item.tokens, candidate))}")
    else:
        lines.append("Edits: none. Reply with an empty edits list.")
    return [ChatMessage(Role.SYSTEM, SYSTEM_PROMPT), ChatMessage(Role.USER, "\n".join(lines))]


def describe_edit(tokens: Sequence[str], candidate: Candidate) -> str:
    """The edit as a reader sees it: what it replaces, inserts or deletes."""
    before = " ".join(tokens[candidate.start : candidate.end])
    after = " ".join(candidate.correction)
    if not before:
        anchor = tokens[candidate.start - 1] if candidate.start > 0 else ""
        where = f'after "{anchor}"' if anchor else "at the start"
        return f'insert "{after}" {where}'
    if not after:
        return f'delete "{before}"'
    return f'replace "{before}" with "{after}"'


def applied(tokens: Sequence[str], candidate: Candidate) -> tuple[str, ...]:
    """``tokens`` with only ``candidate`` applied."""
    return (
        *tokens[: candidate.start],
        *candidate.correction,
        *tokens[candidate.end :],
    )


def judged(reply: JudgeReply, count: int) -> JudgeResult:
    """``reply`` as verdicts in candidate order; ``NotJudged`` unless it rates each edit
    exactly once, since a verdict given to the wrong number would be stored on the wrong edit.
    """
    by_number = {verdict.edit: verdict for verdict in reply.edits}
    if len(by_number) != len(reply.edits) or set(by_number) != set(range(1, count + 1)):
        numbers = sorted(verdict.edit for verdict in reply.edits)
        return NotJudged(f"expected verdicts on edits 1 to {count}, got {numbers}")
    verdicts = tuple((by_number[n].verdict, by_number[n].reason) for n in range(1, count + 1))
    return Judged(verdicts, reply.remaining_error, reply.remaining_reason)
