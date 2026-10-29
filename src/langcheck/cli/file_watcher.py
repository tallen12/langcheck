"""``FileWatcher`` with watchfiles (inotify, FSEvents, or polling as a fallback)."""

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from watchfiles import awatch


class WatchfilesWatcher:
    """Yields after each batch of changes to one file. Editors that save by writing a new
    file and renaming it over the old one are caught because the directory is watched.
    """

    async def changes(self, path: Path) -> AsyncIterator[None]:
        """Yield once per batch of changes that touches ``path``."""
        target = await asyncio.to_thread(path.resolve)
        async for batch in awatch(target.parent):
            changed = [name for _, name in batch]
            if await asyncio.to_thread(touches, changed, target):
                yield None


def touches(changed: Iterable[str], target: Path) -> bool:
    """Whether any changed path is ``target`` (resolving links; blocking)."""
    return any(Path(name).resolve() == target for name in changed)
