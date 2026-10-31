"""Protocols of the context domain."""

from collections.abc import Collection
from typing import Protocol

from langcheck.context.models import DocumentContext
from langcheck.llm.structured import Failed


class ContextSource(Protocol):
    """Describes a whole document for the checker."""

    async def build(
        self, text: str, *, dictionary: Collection[str] = ()
    ) -> DocumentContext | Failed:
        """The context of ``text``, or ``Failed`` when it could not be built."""
        ...
