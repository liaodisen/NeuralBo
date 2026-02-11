from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

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

    @abstractmethod
    def batch(self, role: str, batch_size: int) -> TensorBatch:
        """Return a batch for a given role."""

    def batch_roles(self) -> Iterable[str]:
        """Optional: list the roles this problem supports."""
        return ()

    def validate_role(self, role: str) -> None:
        roles = set(self.batch_roles())
        if roles and role not in roles:
            raise ValueError(f"Unknown batch role: {role}. Expected one of {sorted(roles)}")

    def project_lam(self, lam: torch.Tensor) -> torch.Tensor:
        """Optional projection for lam (e.g. clamp to [0, 1])."""
        return lam


class AIDProblem(NeuralBoProblem, ABC):
    """
    AID-specific extension: exposes inner/outer losses.
    """

    def batch_roles(self) -> Iterable[str]:
        return ("inner", "outer", "hxx", "xw")

    @abstractmethod
    def inner_loss(self, model: nn.Module, lam: torch.Tensor, batch: TensorBatch) -> torch.Tensor:
        ...

    @abstractmethod
    def outer_loss(self, model: nn.Module, lam: torch.Tensor, batch: TensorBatch) -> torch.Tensor:
        ...
