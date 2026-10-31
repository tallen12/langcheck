import io
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

from langcheck.evaluation.datasets import (
    SOURCES,
    Dataset,
    DatasetError,
    Format,
    HttpDownloader,
    dataset_files,
    fetch,
)


def tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def zip_file(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class ServedArchive:
    """A Downloader that writes a prepared archive and counts downloads."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.urls: list[str] = []

    def download(self, url: str, destination: Path) -> None:
        self.urls.append(url)
        destination.write_bytes(self.body)


def test_every_set_has_a_source_and_m2_sets_one_member() -> None:
    assert set(SOURCES) == set(Dataset)
    for dataset, source in SOURCES.items():
        assert source.url.startswith("https://")
        if source.format is Format.M2:
            assert len(source.members) == 1, dataset


def test_fetches_an_m2_set_once_and_keeps_only_its_file(tmp_path: Path) -> None:
    member = SOURCES[Dataset.CONLL14_TEST].members[0]
    archive = ServedArchive(tar_gz({member: b"S a .\n", "other/README": b"x"}))

    files = fetch(Dataset.CONLL14_TEST, tmp_path, downloader=archive)
    again = fetch(Dataset.CONLL14_TEST, tmp_path, downloader=archive)

    assert files == again == dataset_files(Dataset.CONLL14_TEST, tmp_path)
    assert files[0].read_text(encoding="utf-8") == "S a .\n"
    assert sorted(p.name for p in (tmp_path / "conll14-test").iterdir()) == [files[0].name]
    assert len(archive.urls) == 1


def test_fetches_reference_sets_from_zip(tmp_path: Path) -> None:
    members = {name: name.encode() for name in SOURCES[Dataset.JFLEG_DEV].members}

    files = fetch(Dataset.JFLEG_DEV, tmp_path, downloader=ServedArchive(zip_file(members)))

    assert [p.name for p in files] == ["dev.src", "dev.ref0", "dev.ref1", "dev.ref2", "dev.ref3"]


def test_a_member_missing_from_the_archive_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="missing"):
        fetch(Dataset.BEA19_DEV, tmp_path, downloader=ServedArchive(tar_gz({"x": b""})))


def test_http_downloader_streams_to_the_destination(tmp_path: Path) -> None:
    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            return httpx.Response(404)
        return httpx.Response(200, content=b"archive bytes")

    downloader = HttpDownloader(httpx.Client(transport=httpx.MockTransport(serve)))

    downloader.download("https://data.test/set.tar.gz", tmp_path / "set.tar.gz")
    assert (tmp_path / "set.tar.gz").read_bytes() == b"archive bytes"
    with pytest.raises(httpx.HTTPStatusError):
        downloader.download("https://data.test/missing", tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_the_training_sets_read_one_m2_file_each() -> None:
    for dataset in (Dataset.BEA19_TRAIN, Dataset.FCE_TRAIN):
        assert SOURCES[dataset].format is Format.M2
        assert len(SOURCES[dataset].members) == 1
