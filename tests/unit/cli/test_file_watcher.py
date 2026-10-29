import asyncio
from pathlib import Path

from langcheck.cli.file_watcher import WatchfilesWatcher, touches
from langcheck.cli.protocols import FileWatcher


def test_touches_matches_the_resolved_target(tmp_path: Path) -> None:
    target = (tmp_path / "notes.md").resolve()

    assert touches([str(tmp_path / "other.md"), str(tmp_path / "." / "notes.md")], target)
    assert not touches([str(tmp_path / "other.md")], target)


def test_reports_a_change_to_the_watched_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("one", encoding="utf-8")
    watcher: FileWatcher = WatchfilesWatcher()

    async def run() -> None:
        changes = watcher.changes(path)

        async def edit() -> None:
            await asyncio.sleep(0.2)
            await asyncio.to_thread((tmp_path / "other.md").write_text, "x", encoding="utf-8")
            await asyncio.to_thread(path.write_text, "two", encoding="utf-8")

        task = asyncio.create_task(edit())
        await asyncio.wait_for(anext(aiter(changes)), timeout=10)
        await task

    asyncio.run(run())
