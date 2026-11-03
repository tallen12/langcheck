import torch

from langcheck.proposal.storage import StateStore


class Zeros:
    def allocate(self, elements: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(elements, dtype=dtype)


def test_a_store_hands_out_flat_tensors_of_the_asked_size_and_type() -> None:
    store: StateStore = Zeros()

    found = store.allocate(6, torch.float16)

    assert found.shape == (6,)
    assert found.dtype == torch.float16
