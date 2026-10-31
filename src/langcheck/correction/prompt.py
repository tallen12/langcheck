"""The chat sent to the LLM for one flagged sentence (prompt version ``correct-v1``)."""

import json
from collections.abc import Mapping

from langcheck.correction.models import FlaggedSentence, flag_type
from langcheck.correction.schemas import CorrectionReply, CorrectionRequest, RequestFlag
from langcheck.llm.models import ChatMessage, Role
from langcheck.prompts.templates import load_template
from langcheck.tagging.tiers import TextFlag
from langcheck.taxonomy.kinds import ErrorKind

PROMPT_VERSION = "correct-v1"


def flag_ids(flagged: FlaggedSentence) -> dict[str, TextFlag]:
    """The sentence's flags by the IDs the LLM knows them by, in reading order."""
    return {f"F{index}": flag for index, flag in enumerate(flagged.flags, start=1)}


def system_prompt() -> str:
    """The fixed instructions, identical for every sentence so servers can cache them."""
    return load_template(PROMPT_VERSION).substitute(
        kinds="\n".join(f"- {kind.value}" for kind in ErrorKind),
        schema=json.dumps(CorrectionReply.model_json_schema(), indent=2),
    )


def _marked(flagged: FlaggedSentence, ids: Mapping[str, TextFlag]) -> str:
    sentence = flagged.sentence
    offset = sentence.span.start
    parts: list[str] = []
    position = 0
    for name, flag in ids.items():
        start, end = flag.span.start - offset, flag.span.end - offset
        parts.append(sentence.text[position:start])
        if start == end:
            parts.append(f"<{name}/>")
        else:
            parts.append(f"<{name}>{sentence.text[start:end]}</{name}>")
        position = end
    parts.append(sentence.text[position:])
    return "".join(parts)


def correction_request(flagged: FlaggedSentence, *, count: int) -> CorrectionRequest:
    """The structured request for ``flagged``, with offsets into its sentence."""
    sentence = flagged.sentence
    offset = sentence.span.start
    ids = flag_ids(flagged)
    return CorrectionRequest(
        context=sentence.context,
        sentence=sentence.text,
        marked=_marked(flagged, ids),
        corrections_per_flag=count,
        flags=[
            RequestFlag(
                id=name,
                start=flag.span.start - offset,
                end=flag.span.end - offset,
                text=sentence.text[flag.span.start - offset : flag.span.end - offset],
                type=flag_type(flag),
                confidence=flag.tier,
                score=round(flag.score, 3),
            )
            for name, flag in ids.items()
        ],
    )


def correction_messages(flagged: FlaggedSentence, *, count: int) -> list[ChatMessage]:
    """The system and user messages that ask for up to ``count`` corrections per flag."""
    request = correction_request(flagged, count=count)
    return [
        ChatMessage(Role.SYSTEM, system_prompt()),
        ChatMessage(Role.USER, request.model_dump_json(indent=2)),
    ]
