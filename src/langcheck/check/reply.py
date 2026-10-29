"""The schema of the LLM's reply: issues anchored to sentences, never rewritten text.

These are boundary models. ``kind`` stays a plain string here so one issue with an
unknown kind is rejected on its own during resolution instead of failing the whole reply.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MAX_OPTIONS = 4


class ReplyIssue(BaseModel):
    """One problem the model found, quoted exactly as written in one focus sentence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: Annotated[str, Field(min_length=1, description="Sentence ID such as S3.")]
    original: Annotated[
        str, Field(min_length=1, description="The smallest span to change, as written.")
    ]
    occurrence: Annotated[
        int, Field(ge=1, description="Which occurrence of original in the sentence.")
    ] = 1
    kind: Annotated[str, Field(description="An error kind from the taxonomy.")]
    explanation: str
    options: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=MAX_OPTIONS,
            description="Replacements for original, best first.",
        ),
    ]


class CheckReply(BaseModel):
    """The whole reply for one chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issues: list[ReplyIssue]
