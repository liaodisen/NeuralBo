"""
AID-CG (Approximate Implicit Differentiation with Conjugate Gradient), small-batch upper gradient
and
AID-KFAC (AID with KFAC inverse), small-batch upper gradient
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Callable, Any

from src.problem import AIDProblem
from src.solver.utils import conjugate_gradient
from curvlinops import (
    KFACLinearOperator, KFACInverseLinearOperator, FisherType,
)



TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class AIDConfig:
    batch_size: int = 128
    inner_steps: int = 5
    inner_lr: float = 0.001
    outer_lr: float = 1.0
    cg_maxiter: int = 3
    cg_tol: float = 1e-10
    cg_damping: float = 0
    epochs: int = 10


@dataclass
class KFACConfig:
    damping: float = 1e-3
    fisher_type: FisherType = FisherType.MC
    mc_samples: int = 1
    use_heuristic_damping: bool = True
    check_deterministic: bool = False


def _list_dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    """Compute pairwise dot-product sum for two tensor lists.

    Args:
        a: First list of tensors.
        b: Second list of tensors, aligned with `a`.

    Returns:
        A scalar tensor equal to sum_i <a_i, b_i>.
    """
    return sum(torch.sum(x * y) for x, y in zip(a, b))


def _conjugate_gradient(matvec, b, maxiter: int, tol: float, damping: float):
    """Solve a linear system with conjugate gradient using a list-valued matvec.

    Args:
        matvec: Callable that maps a tensor list (and damping) to a tensor list.
        b: Right-hand-side tensor list.
        maxiter: Maximum CG iterations.
        tol: Convergence tolerance.
        damping: Damping factor passed to the matrix-vector product.

    Returns:
        A tensor list approximating the linear solve result.
    """
    def hvp(v_list):
        """Apply damped matrix-vector product.

        Args:
            v_list: Input tensor list.

        Returns:
            Output tensor list after applying `matvec` with damping.
        """
        return matvec(v_list, damping)

    return conjugate_gradient(hvp, b, maxiter=maxiter, tol=tol, lam=damping)


def _next_from_role_iter(problem: AIDProblem, role: str, batch_size: int, role_iters: dict[str, Any]):
    """Fetch the next batch iterator item for a role, restarting on exhaustion.

    Args:
        problem: Problem object providing `iter_batches`.
        role: Batch role key (for example, inner/outer/hxx/xw).
        batch_size: Number of samples per batch.
        role_iters: Mutable mapping from role to active iterator.

    Returns:
        The next batch for `role`.
    """
    it = role_iters.get(role)
    if it is None:
        it = iter(problem.iter_batches(role, batch_size))
    try:
        batch = next(it)
    except StopIteration:
        it = iter(problem.iter_batches(role, batch_size))
        batch = next(it)
    role_iters[role] = it
    return batch


class AIDCGSolver:
    def __init__(
        self,
        config: AIDConfig,
        inner_optimizer: Callable[[List[nn.Parameter]], torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
        outer_optimizer: Callable[[List[nn.Parameter]], torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
    ):
        """Initialize the AID-CG solver configuration and optional optimizers.

        Args:
            config: Solver hyperparameters.
            inner_optimizer: Inner optimizer instance or factory. If `None`, defaults are used.
            outer_optimizer: Outer optimizer instance or factory. If `None`, defaults are used.

        Returns:
            None.
        """
        self.cfg = config
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self._inner_opt = None
        self._outer_opt = None

    def _ensure_optimizers(self, model: nn.Module, lam: torch.Tensor):
        """Create inner/outer optimizers if they have not been initialized.

        Args:
            model: Model containing trainable parameters for inner updates.
            lam: Hyperparameter tensor for outer updates.

        Returns:
            None.
        """
        if self._inner_opt is None:
            if callable(self.inner_optimizer):
                self._inner_opt = self.inner_optimizer([p for p in model.parameters() if p.requires_grad])
            elif self.inner_optimizer is not None:
                self._inner_opt = self.inner_optimizer
            else:
                self._inner_opt = torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=self.cfg.inner_lr,
                    momentum=0.0,
                )
        if self._outer_opt is None:
            if callable(self.outer_optimizer):
                self._outer_opt = self.outer_optimizer([lam])
            elif self.outer_optimizer is not None:
                self._outer_opt = self.outer_optimizer
            else:
                self._outer_opt = torch.optim.SGD([lam], lr=self.cfg.outer_lr, momentum=0.0)

    def step(
        self,
        problem: AIDProblem,
        model: nn.Module,
        lam: torch.Tensor,
        role_iters: dict[str, Any] | None = None,
    ):
        """Run one bilevel optimization step.

        Args:
            problem: Problem instance with inner/outer losses and batch iterators.
            model: Model to optimize.
            lam: Outer variable tensor.
            role_iters: Optional reusable iterators by role.

        Returns:
            None. Updates `model` parameters and `lam` in place.
        """
        cfg = self.cfg
        params = [p for p in model.parameters() if p.requires_grad]
        self._ensure_optimizers(model, lam)
        role_iters = {} if role_iters is None else role_iters

        for _ in range(cfg.inner_steps):
            batch_inner = _next_from_role_iter(problem, "inner", cfg.batch_size, role_iters)
            loss = problem.inner_loss(model, lam, batch_inner)
            self._inner_opt.zero_grad()
            loss.backward()
            self._inner_opt.step()

        batch_outer = _next_from_role_iter(problem, "outer", cfg.batch_size, role_iters)
        f_val = problem.outer_loss(model, lam, batch_outer)
        b = torch.autograd.grad(f_val, params)

        batch_hxx = _next_from_role_iter(problem, "hxx", cfg.batch_size, role_iters)

        def hxx_matvec(v_list, damping):
            """Compute Hessian-vector product for inner-loss parameters.

            Args:
                v_list: Input tensor list.
                damping: Damping factor for regularization.

            Returns:
                Tensor list representing the Hessian-vector product.
            """
            g_val = problem.inner_loss(model, lam, batch_hxx)
            grad_g = torch.autograd.grad(g_val, params, create_graph=True)
            dot_val = _list_dot(grad_g, v_list)
            hv = torch.autograd.grad(dot_val, params)
            if damping and damping > 0:
                hv = [h + damping * v for h, v in zip(hv, v_list)]
            return list(hv)

        s = _conjugate_gradient(
            hxx_matvec,
            list(b),
            maxiter=cfg.cg_maxiter,
            tol=cfg.cg_tol,
            damping=cfg.cg_damping,
        )

        batch_xw = _next_from_role_iter(problem, "xw", cfg.batch_size, role_iters)
        g_xw = problem.inner_loss(model, lam, batch_xw)
        grad_g = torch.autograd.grad(g_xw, params, create_graph=True)
        dot_val = _list_dot(grad_g, s)
        grad_lam = torch.autograd.grad(dot_val, lam)[0]

        self._outer_opt.zero_grad()
        lam.grad = -grad_lam
        self._outer_opt.step()

    def solve(
        self,
        problem: AIDProblem,
        callback: Callable[[int, nn.Module, torch.Tensor, AIDProblem], dict[str, Any] | None] | None = None,
    ):
        """Train the bilevel problem for configured epochs.

        Args:
            problem: Problem instance that provides initialization and losses.
            callback: Optional function called each epoch to log extra metrics.

        Returns:
            A tuple `(model, lam, history)` after optimization.
        """
        model, lam = problem.init()
        history = []
        role_iters: dict[str, Any] = {}
        for epoch in range(self.cfg.epochs):
            self.step(problem, model, lam, role_iters=role_iters)
            record = {"epoch": epoch}
            if callback is not None:
                extra = callback(epoch, model, lam, problem)
                if extra:
                    record.update(extra)
            history.append(record)
        return model, lam, history


class AIDKFACSolver:
    def __init__(
        self,
        config: AIDConfig,
        kfac_cfg: KFACConfig,
        kfac_loss_builder: Callable[[torch.Tensor, TensorBatch], Tuple[nn.Module, torch.Tensor]],
        inner_optimizer: Callable[[List[nn.Parameter]], torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
        outer_optimizer: Callable[[List[nn.Parameter]], torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
    ):
        """Initialize the AID-KFAC solver and its configuration.

        Args:
            config: Solver hyperparameters.
            kfac_cfg: KFAC-specific hyperparameters.
            kfac_loss_builder: Callable that returns `(loss_fn, targets)` for KFAC.
            inner_optimizer: Inner optimizer instance or factory. If `None`, defaults are used.
            outer_optimizer: Outer optimizer instance or factory. If `None`, defaults are used.

        Returns:
            None.
        """
        self.cfg = config
        self.kfac_cfg = kfac_cfg
        self.kfac_loss_builder = kfac_loss_builder
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self._inner_opt = None
        self._outer_opt = None

    def _filter_params(self, model: nn.Module):
        """Collect trainable parameters from Linear/Conv2d modules.

        Args:
            model: Model to scan.

        Returns:
            List of trainable parameters from supported module types.
        """
        params_filtered = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                params_filtered.extend([p for p in m.parameters() if p.requires_grad])
        return params_filtered

    def _ensure_optimizers(self, model: nn.Module, lam: torch.Tensor):
        """Create inner/outer optimizers if they have not been initialized.

        Args:
            model: Model containing trainable parameters for inner updates.
            lam: Hyperparameter tensor for outer updates.

        Returns:
            None.
        """
        if self._inner_opt is None:
            if callable(self.inner_optimizer):
                self._inner_opt = self.inner_optimizer([p for p in model.parameters() if p.requires_grad])
            elif self.inner_optimizer is not None:
                self._inner_opt = self.inner_optimizer
            else:
                self._inner_opt = torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=self.cfg.inner_lr,
                    momentum=0.0,
                )
        if self._outer_opt is None:
            if callable(self.outer_optimizer):
                self._outer_opt = self.outer_optimizer([lam])
            elif self.outer_optimizer is not None:
                self._outer_opt = self.outer_optimizer
            else:
                self._outer_opt = torch.optim.SGD([lam], lr=self.cfg.outer_lr, momentum=0.0)

    def _build_kfac_inverse(
        self,
        model: nn.Module,
        lam: torch.Tensor,
        batch_hxx: TensorBatch,
        params: List[torch.Tensor],
    ):
        """Build a KFAC inverse linear operator for a Hessian/Fisher approximation.

        Args:
            model: Model used to build the linear operator.
            lam: Outer variable tensor passed to the loss builder.
            batch_hxx: Batch used for Hessian/Fisher approximation.
            params: Parameter list defining the operator domain.

        Returns:
            A `KFACInverseLinearOperator` instance.
        """
        loss_fn, targets = self.kfac_loss_builder(lam, batch_hxx)
        x_h = batch_hxx[0]
        hxx = KFACLinearOperator(
            model,
            loss_fn,
            params,
            [(x_h, targets)],
            check_deterministic=self.kfac_cfg.check_deterministic,
            fisher_type=self.kfac_cfg.fisher_type,
            mc_samples=self.kfac_cfg.mc_samples,
        )
        return KFACInverseLinearOperator(
            hxx,
            damping=self.kfac_cfg.damping,
            use_heuristic_damping=self.kfac_cfg.use_heuristic_damping,
        )

    def step(
        self,
        problem: AIDProblem,
        model: nn.Module,
        lam: torch.Tensor,
        role_iters: dict[str, Any] | None = None,
    ):
        """Run one bilevel optimization step using KFAC inverse approximation.

        Args:
            problem: Problem instance with inner/outer losses and batch iterators.
            model: Model to optimize.
            lam: Outer variable tensor.
            role_iters: Optional reusable iterators by role.

        Returns:
            None. Updates `model` parameters and `lam` in place.
        """
        cfg = self.cfg
        params_fil = self._filter_params(model)
        self._ensure_optimizers(model, lam)
        role_iters = {} if role_iters is None else role_iters

        for _ in range(cfg.inner_steps):
            batch_inner = _next_from_role_iter(problem, "inner", cfg.batch_size, role_iters)
            loss = problem.inner_loss(model, lam, batch_inner)
            self._inner_opt.zero_grad()
            loss.backward()
            self._inner_opt.step()

        batch_outer = _next_from_role_iter(problem, "outer", cfg.batch_size, role_iters)
        f_val = problem.outer_loss(model, lam, batch_outer)
        b = torch.autograd.grad(f_val, params_fil)

        batch_hxx = _next_from_role_iter(problem, "hxx", cfg.batch_size, role_iters)
        inv_hxx = self._build_kfac_inverse(model, lam, batch_hxx, params_fil)
        inv_hxx_dx = inv_hxx @ list(b)

        batch_xw = _next_from_role_iter(problem, "xw", cfg.batch_size, role_iters)
        g_xw = problem.inner_loss(model, lam, batch_xw)
        grad_g = torch.autograd.grad(g_xw, params_fil, create_graph=True)
        dot_val = _list_dot(grad_g, inv_hxx_dx)
        grad_lam = torch.autograd.grad(dot_val, lam)[0]

        self._outer_opt.zero_grad()
        lam.grad = -grad_lam
        self._outer_opt.step()

    def solve(
        self,
        problem: AIDProblem,
        callback: Callable[[int, nn.Module, torch.Tensor, AIDProblem], dict[str, Any] | None] | None = None,
    ):
        """Train the bilevel problem for configured epochs using AID-KFAC.

        Args:
            problem: Problem instance that provides initialization and losses.
            callback: Optional function called each epoch to log extra metrics.

        Returns:
            A tuple `(model, lam, history)` after optimization.
        """
        model, lam = problem.init()
        history = []
        role_iters: dict[str, Any] = {}
        for epoch in range(self.cfg.epochs):
            self.step(problem, model, lam, role_iters=role_iters)
            record = {"epoch": epoch}
            if callback is not None:
                extra = callback(epoch, model, lam, problem)
                if extra:
                    record.update(extra)
            history.append(record)
        return model, lam, history
