"""Value types for chat-completion requests, client settings and server replies."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Who wrote a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message of a chat."""

    role: Role
    content: str


type JsonSchema = Mapping[str, object]
"""A JSON Schema document, as plain JSON values."""


@dataclass(frozen=True, slots=True)
class ChatOptions:
    """Sampling and output settings for one request.

    ``json_schema`` asks the server to constrain the reply to that schema (servers that
    cannot still get the instruction in the prompt). ``think=False`` turns off the
    reasoning output of models such as qwen3, which would otherwise spend most of the
    time budget on it, on servers whose ``ClientConfig`` says they take that switch.
    """

    temperature: float = 0.0
    max_tokens: int | None = None
    json_schema: JsonSchema | None = None
    schema_name: str = "reply"
    think: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Where the server is and how hard to try.

    ``base_url`` ends at the API root (for example ``http://localhost:11434/v1``).
    ``api_key`` goes out as ``Authorization: Bearer`` only when set, and is left out of
    ``repr`` so it never reaches a log. Failed requests are retried ``retries`` times,
    waiting ``backoff * 2**attempt`` seconds (at most ``max_backoff``) in between; at
    most ``concurrency`` requests are in flight at once.

    ``reasoning_switch`` says the server takes ``reasoning_effort: "none"`` (Ollama does).
    It is off by default because other servers and models reject that value with a 400,
    which would fail every request.
    """

    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 120.0
    retries: int = 3
    backoff: float = 1.0
    max_backoff: float = 30.0
    concurrency: int = 4
    reasoning_switch: bool = False

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.retries < 0:
            raise ValueError(f"retries must be non-negative, got {self.retries}")
        if self.backoff < 0 or self.max_backoff < 0:
            raise ValueError("backoff and max_backoff must be non-negative")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")


class LLMError(Exception):
    """The server could not produce a reply, even after retries."""


class _ReplyMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    content: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    message: _ReplyMessage
    finish_reason: str | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatCompletion(BaseModel):
    """The part of an OpenAI-compatible ``/chat/completions`` reply this project reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    choices: tuple[_Choice, ...] = Field(min_length=1)
    usage: _Usage | None = None

    @property
    def text(self) -> str:
        """The first choice's text; an empty string when the server sent none."""
        return self.choices[0].message.content or ""
