import math

import torch

from lgdbench import ot


def point_map(grid, r, c):
    m = torch.zeros(1, 1, grid, grid)
    m[0, 0, r, c] = 1.0
    return m


def test_grid_cost_and_target():
    c = ot.grid_cost(4)
    assert c.shape == (16, 16) and c.max() == 1 and c.diag().abs().max() == 0
    t, ok = ot.target_dist(torch.cat([point_map(4, 1, 2), torch.zeros(1, 1, 4, 4)]), 4)
    assert ok.tolist() == [True, False] and torch.isclose(t[0].sum(), torch.tensor(1.0))


def test_sinkhorn_grows_with_distance():
    grid, cost = 8, ot.grid_cost(8)
    b = ot.target_dist(point_map(grid, 2, 2), grid)[0]
    near = ot.log_action_dist(point_map(grid, 2, 3) * 50, grid, 1.0)
    far = ot.log_action_dist(point_map(grid, 7, 7) * 50, grid, 1.0)
    same = ot.log_action_dist(point_map(grid, 2, 2) * 50, grid, 1.0)
    w = [ot.sinkhorn_cost(x, b, cost, 0.01, 100).item() for x in (same, near, far)]
    assert w[0] < w[1] < w[2]
    assert math.isclose(w[2], cost[2 * 8 + 2, 7 * 8 + 7].item(), rel_tol=0.05)


def test_sinkhorn_is_differentiable():
    grid, cost = 6, ot.grid_cost(6)
    pos = torch.rand(2, 1, 6, 6, requires_grad=True)
    b = ot.target_dist(torch.rand(2, 1, 6, 6), grid)[0]
    ot.sinkhorn_cost(ot.log_action_dist(pos, grid, 0.5), b, cost, 0.1, 20).sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0


def test_margin():
    assert ot.margin(torch.tensor(0.2), torch.tensor(0.5), 0.1).item() == 0.0
    assert math.isclose(ot.margin(torch.tensor(0.5), torch.tensor(0.5), 0.1).item(), 0.1, rel_tol=1e-6)


def test_transport_terms_prefer_own_region_and_margin_hinge():
    own = torch.zeros(1, 1, 56, 56)
    own[..., 5:10, 5:10] = 1.0
    other = torch.zeros(1, 1, 56, 56)
    other[..., 45:50, 45:50] = 1.0
    pull_near, push_near = ot.transport_terms(own * 5, own, other, 0.1, 0.05, 0.05, 50)     # mass on its own region
    pull_far, push_far = ot.transport_terms(other * 5, own, other, 0.1, 0.05, 0.05, 50)     # mass on the other object
    assert pull_near < pull_far
    assert push_near == 0 and push_far > 0
    pull_none, push_none = ot.transport_terms(own * 5, own, torch.zeros_like(own), 0.1, 0.05, 0.05, 50)  # no other object
    assert push_none == 0 and torch.isclose(pull_none, pull_near)
