"""State stores in memory and in a memory-mapped file."""

from pathlib import Path

import torch


class MemoryStore:
    """Keeps states in RAM, for sets small enough to fit."""

    def allocate(self, elements: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(elements, dtype=dtype)


class MappedFileStore:
    """Keeps states in files under ``folder``, so the page cache holds what fits in RAM and
    the rest is read back from disk. The caller owns ``folder`` and removes it afterwards.
    """

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._count = 0

    def allocate(self, elements: int, dtype: torch.dtype) -> torch.Tensor:
        self._count += 1
        path = self._folder / f"states-{self._count}.bin"
        size = elements * torch.empty((), dtype=dtype).element_size()
        with path.open("wb") as file:
            file.truncate(size)
        return torch.from_file(str(path), shared=True, size=elements, dtype=dtype)
