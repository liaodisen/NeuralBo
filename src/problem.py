from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

import torch
import torch.nn as nn


TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class NeuralBoProblem(ABC):
    """
    Base class for neural bilevel optimization problems.

    Subclass this and implement the abstract methods. Concrete solvers depend on
    extensions of this base (e.g., AIDProblem).
    """

    @abstractmethod
    def init(self) -> tuple[nn.Module, torch.Tensor]:
        """Return (model, lam)."""

    def batch(self, role: str, batch_size: int) -> Any:
        """
        Return a single batch for a given role.

        Override this method for random-access sampling.
        """
        raise NotImplementedError("Override batch(...) or dataloader(...) in your problem.")

    def dataloader(self, role: str, batch_size: int) -> Iterable[Any] | None:
        """
        Optional role-specific iterable (e.g. torch DataLoader or generator).

        If provided, solvers consume this iterable directly. Otherwise, solvers
        will fall back to repeatedly calling batch(...).
        """
        return None

    def iter_batches(self, role: str, batch_size: int) -> Iterable[Any]:
        """Yield batches for a role from a dataloader or from repeated sampling."""
        self.validate_role(role)
        loader = self.dataloader(role, batch_size)
        if loader is not None:
            return loader

        def _infinite_sampler():
            while True:
                yield self.batch(role, batch_size)

        return _infinite_sampler()

    def batch_roles(self) -> Iterable[str]:
        """Optional: list the roles this problem supports."""
        return ()

    def validate_role(self, role: str) -> None:
        roles = set(self.batch_roles())
        if roles and role not in roles:
            raise ValueError(f"Unknown batch role: {role}. Expected one of {sorted(roles)}")


class AIDProblem(NeuralBoProblem, ABC):
    """
    AID-specific extension: exposes inner/outer losses.
    """

    def batch_roles(self) -> Iterable[str]:
        return ("inner", "outer", "hxx", "xw")

    @abstractmethod
    def inner_loss(self, model: nn.Module, lam: torch.Tensor, batch: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def outer_loss(self, model: nn.Module, lam: torch.Tensor, batch: Any) -> torch.Tensor:
        ...
