"""What the LLM is sent about a sentence's flags, and the schema of its reply.

These are boundary models. ``kind`` stays a plain string in the reply so one correction
with an unknown kind is rejected on its own instead of failing the whole reply.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from langcheck.correction.models import FlagType
from langcheck.tagging.tiers import Tier


class RequestFlag(BaseModel):
    """One suspected error, with offsets into the request's sentence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    start: int
    end: int
    text: str
    type: FlagType
    confidence: Tier
    score: float


class CorrectionRequest(BaseModel):
    """One sentence and its flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: str
    sentence: str
    marked: str
    corrections_per_flag: int
    flags: list[RequestFlag]


class ReplyCorrection(BaseModel):
    """One proposed edit near a flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original: Annotated[
        str,
        Field(description="The text to change, as written; empty for exactly the flagged text."),
    ] = ""
    replacement: Annotated[str, Field(description="The new text; empty to delete.")]
    kind: Annotated[str, Field(description="An error kind from the taxonomy.")]
    explanation: str


class ReplyFlag(BaseModel):
    """The corrections for one flag, best first; none when the text there is correct."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flag: Annotated[str, Field(min_length=1, description="Flag ID such as F1.")]
    corrections: list[ReplyCorrection]


class CorrectionReply(BaseModel):
    """The whole reply for one sentence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flags: list[ReplyFlag]
