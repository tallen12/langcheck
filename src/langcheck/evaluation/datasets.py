"""The annotated sets the plan evaluates and trains on, and fetching them.

Files are kept under ``data/eval/<name>/`` (``/data/`` is not committed). Each set is
downloaded once, as published, and only the files it needs are unpacked.
"""

import tarfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

import httpx


class Dataset(StrEnum):
    """The evaluation sets."""

    BEA19_DEV = "bea19-dev"
    CONLL14_TEST = "conll14-test"
    JFLEG_DEV = "jfleg-dev"
    BEA19_TRAIN = "bea19-train"
    FCE_TRAIN = "fce-train"


class Format(StrEnum):
    """How a set's gold corrections are written."""

    M2 = "m2"
    """Gold edits, typed by ERRANT, from one or more annotators."""
    REFERENCES = "references"
    """Tokenized sources and whole corrected references; edits come from ERRANT."""


@dataclass(frozen=True, slots=True)
class Source:
    """Where a set is published and which archive members it uses.

    For ``M2`` the one member is the M2 file; for ``REFERENCES`` the first member is the
    source sentences and the rest one reference each, line by line.
    """

    url: str
    members: tuple[str, ...]
    format: Format
    licence: str


SOURCES: dict[Dataset, Source] = {
    Dataset.BEA19_DEV: Source(
        url="https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz",
        members=("wi+locness/m2/ABCN.dev.gold.bea19.m2",),
        format=Format.M2,
        licence="W&I+LOCNESS v2.1, for research use (see the BEA-2019 shared task site)",
    ),
    Dataset.CONLL14_TEST: Source(
        url="https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz",
        members=("conll14st-test-data/noalt/official-2014.combined.m2",),
        format=Format.M2,
        licence="CoNLL-2014 shared task test data (NUS), for research use",
    ),
    Dataset.JFLEG_DEV: Source(
        url="https://github.com/keisks/jfleg/archive/refs/heads/master.zip",
        members=(
            "jfleg-master/dev/dev.src",
            "jfleg-master/dev/dev.ref0",
            "jfleg-master/dev/dev.ref1",
            "jfleg-master/dev/dev.ref2",
            "jfleg-master/dev/dev.ref3",
        ),
        format=Format.REFERENCES,
        licence="JFLEG, CC BY-NC-SA 4.0",
    ),
    Dataset.BEA19_TRAIN: Source(
        url="https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz",
        members=("wi+locness/m2/ABC.train.gold.bea19.m2",),
        format=Format.M2,
        licence="W&I+LOCNESS v2.1, for research use (see the BEA-2019 shared task site)",
    ),
    Dataset.FCE_TRAIN: Source(
        url="https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz",
        members=("fce/m2/fce.train.gold.bea19.m2",),
        format=Format.M2,
        licence="FCE v2.1, for research use (see the BEA-2019 shared task site)",
    ),
}


class Downloader(Protocol):
    """Saves the resource at a URL to a file."""

    def download(self, url: str, destination: Path) -> None:
        """Write the body at ``url`` to ``destination``."""
        ...


class DatasetError(Exception):
    """A set's archive does not hold the files expected."""


def dataset_files(dataset: Dataset, directory: Path) -> list[Path]:
    """Where the set's files are kept once fetched, in ``Source.members`` order."""
    return [directory / dataset.value / PurePosixPath(m).name for m in SOURCES[dataset].members]


def fetch(dataset: Dataset, directory: Path, *, downloader: Downloader) -> list[Path]:
    """Download and unpack the set unless it is there already; its files."""
    files = dataset_files(dataset, directory)
    if all(path.is_file() for path in files):
        return files
    source = SOURCES[dataset]
    target = directory / dataset.value
    target.mkdir(parents=True, exist_ok=True)
    archive = target / PurePosixPath(source.url).name
    downloader.download(source.url, archive)
    _unpack(archive, source.members, target)
    archive.unlink()
    return files


def _unpack(archive: Path, members: tuple[str, ...], target: Path) -> None:
    """Copy just ``members`` out of a ``.tar.gz`` or ``.zip``, flat into ``target``."""
    wanted = set(members)
    found: set[str] = set()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as package:
            for name in package.namelist():
                if name in wanted:
                    (target / PurePosixPath(name).name).write_bytes(package.read(name))
                    found.add(name)
    else:
        with tarfile.open(archive, "r:gz") as package:
            for member in package.getmembers():
                if member.name in wanted and member.isfile():
                    extracted = package.extractfile(member)
                    if extracted is not None:
                        (target / PurePosixPath(member.name).name).write_bytes(extracted.read())
                        found.add(member.name)
    missing = wanted - found
    if missing:
        msg = f"{archive.name} is missing {sorted(missing)}"
        raise DatasetError(msg)


class HttpDownloader:
    """Streams downloads with httpx, writing to a temporary name until complete."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def download(self, url: str, destination: Path) -> None:
        """Write the body at ``url`` to ``destination``; raises ``httpx.HTTPError``."""
        partial = destination.with_name(destination.name + ".part")
        with self._http.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with partial.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        partial.replace(destination)
