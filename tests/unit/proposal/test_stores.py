from pathlib import Path

import torch

from langcheck.proposal.stores import MappedFileStore, MemoryStore


def test_memory_store_hands_out_writable_tensors() -> None:
    found = MemoryStore().allocate(4, torch.float16)
    found.copy_(torch.arange(4))

    assert found.dtype == torch.float16
    assert found.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_mapped_store_writes_through_to_a_file_per_allocation(tmp_path: Path) -> None:
    store = MappedFileStore(tmp_path)

    first = store.allocate(4, torch.float16)
    second = store.allocate(2, torch.float16)
    first.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    second.fill_(7.0)

    files = sorted(tmp_path.iterdir())
    assert [f.stat().st_size for f in files] == [8, 4]
    reread = torch.from_file(str(files[0]), shared=False, size=4, dtype=torch.float16)
    assert reread.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert second.tolist() == [7.0, 7.0]
