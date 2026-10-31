"""Optimal-transport loss of GRAT: W+ pulls the predicted grasp mass to the instruction's grasps, W- keeps a margin to
another object's grasps. Both are Sinkhorn distances on a 28 x 28 grid."""
from __future__ import annotations

import torch
import torch.nn.functional as F

GRID = 28


def log_action_dist(pos: torch.Tensor, grid: int, tau: float) -> torch.Tensor:
    """B x grid^2 log-probabilities of the action distribution softmax(avg-pooled pos / tau)."""
    return F.log_softmax(F.adaptive_avg_pool2d(pos, grid).flatten(1) / tau, dim=1)


def target_dist(qmap: torch.Tensor, grid: int):
    """GT quality map pooled to grid^2 and normalised; ok (B,) False for empty maps."""
    t = F.adaptive_avg_pool2d(qmap, grid).flatten(1)
    s = t.sum(1, keepdim=True)
    return t / s.clamp_min(1e-12), s[:, 0] > 0


def grid_cost(grid: int, device=None) -> torch.Tensor:
    """grid^2 x grid^2 squared distances between cell centres, divided by the largest one."""
    r = torch.arange(grid, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    p = torch.stack([yy.flatten(), xx.flatten()], 1)
    c = torch.cdist(p, p) ** 2
    return c / c.max()


def sinkhorn_cost(log_a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
    """Entropic OT cost <P, C> per batch item between exp(log_a) (B x N, differentiable) and b (B x N, a distribution),
    log-domain Sinkhorn. Cells with zero target mass are handled by a -inf log."""
    log_b = torch.log(b.clamp_min(0)).masked_fill(b <= 0, float("-inf"))
    k = -cost / eps                                                   # N x N
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_a)
    for _ in range(iters):
        f = log_a - torch.logsumexp(k[None] + g[:, None, :], dim=2)
        g = log_b - torch.logsumexp(k[None] + f[:, :, None], dim=1)
        g = torch.where(torch.isfinite(g), g, torch.full_like(g, -1e9))
    log_p = k[None] + f[:, :, None] + g[:, None, :]
    return (log_p.exp() * cost[None]).sum(dim=(1, 2))


def margin(own: torch.Tensor, other: torch.Tensor, m: float) -> torch.Tensor:
    """Hinge: the distance to the instruction's own grasps must be smaller by m than to another object's."""
    return F.relu(m + own - other)


def transport_terms(x1_hat, y_pos, other_pos, tau: float, m: float, eps: float, iters: int):
    """(pull, push): mean W+ and mean hinge on W+ and W-, for the quality channel ``x1_hat`` of the one-step endpoint.
    Samples without another object (empty ``other_pos``) contribute to the pull only."""
    cost = grid_cost(GRID, x1_hat.device)
    log_p = log_action_dist(x1_hat, GRID, tau)
    q, ok = target_dist(y_pos, GRID)
    q2, ok2 = target_dist(other_pos, GRID)
    own = sinkhorn_cost(log_p, q, cost, eps, iters)
    oth = sinkhorn_cost(log_p, torch.where(ok2[:, None], q2, q), cost, eps, iters)
    pull = own[ok].mean() if ok.any() else own.sum() * 0
    both = ok & ok2
    push = margin(own, oth, m)[both].mean() if both.any() else own.sum() * 0
    return pull, push
