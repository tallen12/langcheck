import asyncio
from collections.abc import Collection

from langcheck.context.models import DocumentContext
from langcheck.context.protocols import ContextSource
from langcheck.llm.structured import Failed


class EmptyContexts:
    """Describes every document as saying nothing."""

    async def build(
        self, text: str, *, dictionary: Collection[str] = ()
    ) -> DocumentContext | Failed:
        return DocumentContext.empty()


def test_context_source_contract() -> None:
    source: ContextSource = EmptyContexts()

    assert asyncio.run(source.build("Hi.")) == DocumentContext.empty()
