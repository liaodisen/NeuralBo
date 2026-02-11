import math
from typing import List

import torch


def _norm(tensors: List[torch.Tensor]) -> float:
    return math.sqrt(sum([torch.sum(tensor ** 2).item() for tensor in tensors]))


def _dot(tensors_one: List[torch.Tensor], tensors_two: List[torch.Tensor]) -> torch.Tensor:
    ret = tensors_one[0].new_zeros((1,), requires_grad=True)
    for t1, t2 in zip(tensors_one, tensors_two):
        ret = ret + torch.sum(t1 * t2)
    return ret


@torch.no_grad()
def conjugate_gradient(_hvp, b, maxiter=None, tol=1e-8, lam=0.0, use_cache=0):
    """
    Minimize 0.5 x^T H^T H x - b^T H x, where H is symmetric
    Args:
        _hvp (function): hessian vector product, only takes a sequence of tensors as input
        b (sequence of tensors): b
        maxiter (int): number of iterations
        tol (float): tolerance for convergence
        lam (float): regularization constant to avoid singularity of hessian
    Returns:
        xxs: solution to the minimization problem
    """
    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)
        outputs = [xx + lam * yy for xx, yy in zip(outputs, inputs)]
        return outputs

    with torch.enable_grad():
        Hb = hvp(b)

    # zero initialization
    xxs = [hb.new_zeros(hb.size()) for hb in Hb]
    ggs = [-hb for hb in Hb]  # Remove unnecessary clone().detach()
    dds = [-hb for hb in Hb]  # Remove unnecessary clone().detach()

    i = 0
    prev_residual_norm = float("inf")

    while True:
        i += 1

        with torch.enable_grad():
            Qdds = hvp(hvp(dds))

        # Check convergence
        residual_norm = _norm(ggs)
        if residual_norm < tol:
            break

        # Check if progress is too small (stagnation)
        if abs(prev_residual_norm - residual_norm) < tol * 1e-3:
            break

        prev_residual_norm = residual_norm

        # Compute step size with safety check
        dQd = _dot(dds, Qdds)
        if abs(dQd) < 1e-12:  # Avoid division by very small numbers
            break

        alpha = -_dot(dds, ggs) / dQd
        xxs = [xx + alpha * dd for xx, dd in zip(xxs, dds)]

        # Update gradient
        ggs = [gg + alpha * Qdd for gg, Qdd in zip(ggs, Qdds)]

        # Compute the next conjugate direction with safety check
        beta = _dot(ggs, Qdds) / dQd
        dds = [gg - beta * dd for gg, dd in zip(ggs, dds)]

        if maxiter is not None and i >= maxiter:
            break

    return xxs
