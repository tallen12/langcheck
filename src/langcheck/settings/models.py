"""Nested settings sections, each a frozen Pydantic model.

Domain code never sees these; the composition roots (CLI and API startup) convert them to
the domain's own values, for example :meth:`LLMSettings.client_config`.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langcheck.check.chunk import DEFAULT_MAX_WORDS
from langcheck.context.builder import DEFAULT_MAX_PARAGRAPHS
from langcheck.documents.sessions import DEFAULT_DEBOUNCE, DEFAULT_REFRESH_SHARE
from langcheck.llm.models import ClientConfig
from langcheck.scoring.factory import ScorerAdapter, ScorerConfig
from langcheck.scoring.policy import KindPolicy, ScoringPolicy
from langcheck.scoring.threaded import DEFAULT_MAX_BATCH
from langcheck.taxonomy.kinds import ErrorKind

_SECTION = ConfigDict(frozen=True, extra="forbid")


class HttpSettings(BaseModel):
    """Where ``langcheck serve`` listens. TLS belongs to a reverse proxy in front."""

    model_config = _SECTION

    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000


class LLMSettings(BaseModel):
    """The OpenAI-compatible LLM server."""

    model_config = _SECTION

    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:8b"
    api_key: SecretStr | None = None
    timeout: Annotated[float, Field(gt=0)] = 120.0
    retries: Annotated[int, Field(ge=0)] = 3
    concurrency: Annotated[int, Field(ge=1)] = 4
    reasoning_switch: bool = False

    def client_config(self) -> ClientConfig:
        """The LLM client's configuration."""
        return ClientConfig(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key is not None else None,
            timeout=self.timeout,
            retries=self.retries,
            concurrency=self.concurrency,
            reasoning_switch=self.reasoning_switch,
        )


class ServerConnection(BaseModel):
    """How the CLI reaches a running ``langcheck serve``. No URL means check locally."""

    model_config = _SECTION

    url: str | None = None
    api_key: SecretStr | None = None
    timeout: Annotated[float, Field(gt=0)] = 300.0


class AuthMode(StrEnum):
    """How the server authenticates requests."""

    API_KEY = "api_key"
    NONE = "none"


class AuthSettings(BaseModel):
    """``api_key`` (the default) needs a bearer token per user. ``none`` trusts every
    request, as one anonymous user or as the user a reverse proxy names in
    ``proxy_user_header``.
    """

    model_config = _SECTION

    mode: AuthMode = AuthMode.API_KEY
    proxy_user_header: str | None = None


class DatabaseSettings(BaseModel):
    """The SQL database for API keys (and, later, dictionaries)."""

    model_config = _SECTION

    url: str = "sqlite:///langcheck.db"


class CheckerSettings(BaseModel):
    """How text is split up for the LLM, and how open documents are re-checked:
    ``debounce`` seconds after the last edit, rebuilding the document context once more
    than ``context_refresh_share`` of its sentences changed.
    """

    model_config = _SECTION

    max_words: Annotated[int, Field(ge=1)] = DEFAULT_MAX_WORDS
    context_max_paragraphs: Annotated[int, Field(ge=3)] = DEFAULT_MAX_PARAGRAPHS
    debounce: Annotated[float, Field(ge=0)] = DEFAULT_DEBOUNCE
    context_refresh_share: Annotated[float, Field(ge=0, le=1)] = DEFAULT_REFRESH_SHARE


class KindPolicySettings(BaseModel):
    """Filtering for one error kind: drop an issue unless an option beats the original
    sentence by more than ``margin``.
    """

    model_config = _SECTION

    filter: bool = False
    margin: float = 0.0


class ScorerSettings(BaseModel):
    """Which sentence scorers rank options. No adapters (the default) means options keep
    the LLM's order and PyTorch is not needed; ``pll`` and ``cola`` need a PyTorch build
    (``just setup cpu``) and download their models on first use.
    """

    model_config = _SECTION

    adapters: list[ScorerAdapter] = Field(default_factory=list)
    device: str = "cpu"
    pll_model: str = "distilroberta-base"
    cola_model: str = "textattack/roberta-base-CoLA"
    window: Annotated[int, Field(ge=0)] = 3
    max_batch: Annotated[int, Field(ge=1)] = DEFAULT_MAX_BATCH
    policy: dict[ErrorKind, KindPolicySettings] = Field(default_factory=dict)

    def scorer_config(self) -> ScorerConfig:
        """The scoring domain's configuration."""
        return ScorerConfig(
            adapters=tuple(dict.fromkeys(self.adapters)),
            device=self.device,
            pll_model=self.pll_model,
            cola_model=self.cola_model,
            window=self.window,
            max_batch=self.max_batch,
            policy=ScoringPolicy(
                {kind: KindPolicy(rule.filter, rule.margin) for kind, rule in self.policy.items()}
            ),
        )


class LimitsSettings(BaseModel):
    """Per-user limits on checks (one-shot or in a session), so one user cannot starve
    the others: how many may run at once and how many may start per minute.
    """

    model_config = _SECTION

    in_flight: Annotated[int, Field(ge=1)] = 2
    per_minute: Annotated[int, Field(ge=1)] = 30
