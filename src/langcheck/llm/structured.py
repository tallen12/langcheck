"""Asking for a JSON reply, validating it with Pydantic and retrying once on failure.

Servers differ in whether they honour ``response_format``, so the schema is sent as
``json_schema`` and is expected in the prompt as well; every reply is validated, and an
invalid one is retried once with the validation error appended to the chat.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydantic import BaseModel, ValidationError

from langcheck.llm.models import ChatMessage, ChatOptions, LLMError, Role
from langcheck.llm.protocols import AsyncLLMClient

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Parsed[T: BaseModel]:
    """A reply that validated."""

    value: T


@dataclass(frozen=True, slots=True)
class Failed:
    """No valid reply: the server failed, or both attempts were invalid."""

    reason: str


type StructuredResult[T: BaseModel] = Parsed[T] | Failed


async def complete_structured[T: BaseModel](
    llm: AsyncLLMClient,
    messages: Sequence[ChatMessage],
    model: type[T],
    *,
    options: ChatOptions,
) -> StructuredResult[T]:
    """Ask ``llm`` for a reply matching ``model``, retrying once if it does not validate."""
    options = replace(options, json_schema=model.model_json_schema())
    chat = list(messages)
    reason = "no attempt made"
    for _ in range(2):
        try:
            reply = await llm.complete(chat, options=options)
        except LLMError as error:
            return Failed(f"LLM error: {error}")
        try:
            return Parsed(model.model_validate_json(strip_fence(reply)))
        except ValidationError as error:
            reason = f"invalid reply: {error}"
            chat += [
                ChatMessage(Role.ASSISTANT, reply),
                ChatMessage(
                    Role.USER,
                    "Your reply did not match the schema:\n"
                    f"{error}\n"
                    "Reply again with only the corrected JSON object.",
                ),
            ]
    return Failed(reason)


def strip_fence(reply: str) -> str:
    """The reply without a surrounding Markdown code fence, which some models add."""
    match = _FENCE.match(reply)
    if match is None:
        return reply
    return reply[match.start(1) : match.end(1)]
