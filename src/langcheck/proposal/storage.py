"""Where training keeps the frozen encoder's states between epochs."""

from typing import Protocol

import torch


class StateStore(Protocol):
    """Hands out flat CPU tensors that stay valid for as long as the store does."""

    def allocate(self, elements: int, dtype: torch.dtype) -> torch.Tensor:
        """A writable one-dimensional tensor of ``elements`` values, contents undefined."""
        ...
