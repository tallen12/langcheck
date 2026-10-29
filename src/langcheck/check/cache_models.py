"""Keys and values of the per-user result cache."""

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass, replace

from langcheck.check.suggestion import Suggestion, suggestion_id
from langcheck.text.models import Span


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Identifies one sentence checked in one setting, for one user."""

    user: str
    digest: str


@dataclass(frozen=True, slots=True)
class CacheSetting:
    """Everything besides the text that changes what the LLM would say."""

    model: str
    prompt_version: str
    context_version: str
    ranker: str


def cache_key(
    *,
    user: str,
    sentence: str,
    previous: str,
    following: str,
    setting: CacheSetting,
) -> CacheKey:
    """The key for ``sentence`` between its neighbouring sentences, in ``setting``."""
    fields = [
        sentence,
        previous,
        following,
        setting.model,
        setting.prompt_version,
        setting.context_version,
        setting.ranker,
    ]
    encoded = json.dumps(fields, ensure_ascii=False).encode("utf-8")
    return CacheKey(user, hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class SentenceResult:
    """The suggestions for one sentence, with spans relative to the sentence start."""

    suggestions: tuple[Suggestion, ...]

    @classmethod
    def relative_to(cls, start: int, suggestions: Collection[Suggestion]) -> SentenceResult:
        """Store ``suggestions`` found in a sentence that starts at ``start``."""
        return cls(tuple(move(s, -start) for s in suggestions))

    def placed_at(self, start: int) -> tuple[Suggestion, ...]:
        """The suggestions for the same sentence now starting at ``start``."""
        return tuple(move(s, start) for s in self.suggestions)


def move(suggestion: Suggestion, offset: int) -> Suggestion:
    """``suggestion`` shifted by ``offset`` code points, with the ID for its new span."""
    span = Span(suggestion.span.start + offset, suggestion.span.end + offset)
    return replace(
        suggestion,
        span=span,
        id=suggestion_id(span, suggestion.original, suggestion.kind),
    )
