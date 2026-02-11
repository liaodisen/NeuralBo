"""
AID-CG (Approximate Implicit Differentiation with Conjugate Gradient), small-batch upper gradient
and
AID-KFAC (AID with KFAC inverse), small-batch upper gradient
"""

import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import List, Tuple, Callable, Any

from src.problem import AIDProblem
from src.solver.utils import conjugate_gradient
from curvlinops import (
    KFACLinearOperator, KFACInverseLinearOperator, FisherType, EKFACLinearOperator,
    HessianLinearOperator, GGNLinearOperator, CGInverseLinearOperator, NeumannInverseLinearOperator,
)
from backpack.hessianfree.ggnvp import ggn_vector_product
from curvlinops.weighted_ce_loss import CrossEntropyLossWeighted
from linear_operator.utils.linear_cg import linear_cg



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
    return sum(torch.sum(x * y) for x, y in zip(a, b))


def _list_add(a: List[torch.Tensor], b: List[torch.Tensor], alpha: float = 1.0):
    return [x + alpha * y for x, y in zip(a, b)]


def _list_sub(a: List[torch.Tensor], b: List[torch.Tensor]):
    return [x - y for x, y in zip(a, b)]


def _list_zeros_like(a: List[torch.Tensor]):
    return [torch.zeros_like(x) for x in a]


def _conjugate_gradient(matvec, b, maxiter: int, tol: float, damping: float):
    def hvp(v_list):
        return matvec(v_list, damping)

    return conjugate_gradient(hvp, b, maxiter=maxiter, tol=tol, lam=damping)


class AIDCGSolver:
    def __init__(
        self,
        config: AIDConfig,
        inner_optimizer: Callable[[List[nn.Parameter]], torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
        outer_optimizer: Callable[[List[nn.Parameter]], torch.optim.Optimizer] | torch.optim.Optimizer | None = None,
    ):
        self.cfg = config
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self._inner_opt = None
        self._outer_opt = None

    def _ensure_optimizers(self, model: nn.Module, lam: torch.Tensor):
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

    def step(self, problem: AIDProblem, model: nn.Module, lam: torch.Tensor):
        cfg = self.cfg
        params = [p for p in model.parameters() if p.requires_grad]
        self._ensure_optimizers(model, lam)

        for _ in range(cfg.inner_steps):
            batch_inner = problem.batch("inner", cfg.batch_size)
            loss = problem.inner_loss(model, lam, batch_inner)
            self._inner_opt.zero_grad()
            loss.backward()
            self._inner_opt.step()

        batch_outer = problem.batch("outer", cfg.batch_size)
        f_val = problem.outer_loss(model, lam, batch_outer)
        b = torch.autograd.grad(f_val, params)

        batch_hxx = problem.batch("hxx", cfg.batch_size)

        def hxx_matvec(v_list, damping):
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

        batch_xw = problem.batch("xw", cfg.batch_size)
        g_xw = problem.inner_loss(model, lam, batch_xw)
        grad_g = torch.autograd.grad(g_xw, params, create_graph=True)
        dot_val = _list_dot(grad_g, s)
        grad_lam = torch.autograd.grad(dot_val, lam)[0]

        self._outer_opt.zero_grad()
        lam.grad = -grad_lam
        self._outer_opt.step()
        with torch.no_grad():
            lam.copy_(problem.project_lam(lam))

    def solve(self, problem: AIDProblem):
        model, lam = problem.init()
        history = []
        for epoch in range(self.cfg.epochs):
            self.step(problem, model, lam)
            history.append({"epoch": epoch})
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
        self.cfg = config
        self.kfac_cfg = kfac_cfg
        self.kfac_loss_builder = kfac_loss_builder
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self._inner_opt = None
        self._outer_opt = None

    def _filter_params(self, model: nn.Module):
        """Ensure only nn.Linear and nn.Conv are used"""
        params_filtered = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                params_filtered.extend([p for p in m.parameters() if p.requires_grad])
        return params_filtered

    def _ensure_optimizers(self, model: nn.Module, lam: torch.Tensor):
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
        loss_fn, targets = self.kfac_loss_builder(lam, batch_hxx)
        x_h, _, _ = batch_hxx
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

    def step(self, problem: AIDProblem, model: nn.Module, lam: torch.Tensor):
        cfg = self.cfg
        params_fil = self._filter_params(model)
        self._ensure_optimizers(model, lam)

        for _ in range(cfg.inner_steps):
            batch_inner = problem.batch("inner", cfg.batch_size)
            loss = problem.inner_loss(model, lam, batch_inner)
            self._inner_opt.zero_grad()
            loss.backward()
            self._inner_opt.step()

        batch_outer = problem.batch("outer", cfg.batch_size)
        f_val = problem.outer_loss(model, lam, batch_outer)
        b = torch.autograd.grad(f_val, params_fil)

        batch_hxx = problem.batch("hxx", cfg.batch_size)
        inv_hxx = self._build_kfac_inverse(model, lam, batch_hxx, params_fil)
        inv_hxx_dx = inv_hxx @ list(b)

        batch_xw = problem.batch("xw", cfg.batch_size)
        g_xw = problem.inner_loss(model, lam, batch_xw)
        grad_g = torch.autograd.grad(g_xw, params_fil, create_graph=True)
        dot_val = _list_dot(grad_g, inv_hxx_dx)
        grad_lam = torch.autograd.grad(dot_val, lam)[0]

        self._outer_opt.zero_grad()
        lam.grad = -grad_lam
        self._outer_opt.step()
        with torch.no_grad():
            lam.copy_(problem.project_lam(lam))

    def solve(self, problem: AIDProblem):
        model, lam = problem.init()
        history = []
        for epoch in range(self.cfg.epochs):
            self.step(problem, model, lam)
            history.append({"epoch": epoch})
        return model, lam, history
