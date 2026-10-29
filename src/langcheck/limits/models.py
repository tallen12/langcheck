"""The outcomes of asking for a check slot."""

from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self


@dataclass(slots=True)
class Permit:
    """A check slot; leaving the ``with`` block (or ``release``) frees it once."""

    _release: Callable[[], None] = field(repr=False)
    _released: bool = False

    def release(self) -> None:
        """Free the slot; later calls do nothing."""
        if not self._released:
            self._released = True
            self._release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class RateLimited:
    """No slot now; try again after ``retry_after`` seconds."""

    retry_after: float


type Admission = Permit | RateLimited
