import dataclasses

import pytest
import torch

pytest.importorskip("clip")
from lgdbench import grat  # noqa: E402

CFG = grat.Config(pose_prior=(0.1, -0.05, 0.3))


def test_config_validates():
    assert not grat.Config().uses_ot and grat.Config(lam=20.0).uses_ot
    for bad in ({"source": "x"}, {"lam": -1.0}, {"steps": 0}):
        with pytest.raises(ValueError):
            grat.Config(**bad)


def test_source_constant_channels():
    rel = torch.rand(2, 1, 4, 4)
    cfg = dataclasses.replace(CFG, source="constant")
    x0 = grat.source_maps(rel, cfg, sample=False)
    assert x0.shape == (2, 4, 4, 4)
    assert torch.equal(x0[:, :1], rel)                       # quality: the instruction's relevance map
    for i, v in enumerate((0.1, -0.05, 0.3)):                # cos, sin, width: the constant prior
        assert torch.allclose(x0[:, i + 1], torch.full((2, 4, 4), v))
    torch.manual_seed(0)
    assert 0 < (grat.source_maps(rel, cfg, sample=True) - x0).abs().max() < 1   # + 0.1 noise on every channel in training


def test_source_relevance_mixes_prior_and_background():
    rel = torch.tensor([[[[0.0, 1.0], [0.5, 0.25]]]])
    x0 = grat.source_maps(rel, CFG, sample=False)
    assert torch.equal(x0[:, :1], rel)
    assert torch.allclose(x0[:, 1:, 0, 0], torch.tensor([[1.0, 0.0, 0.0]]))     # r = 0: the label value outside grasps
    assert torch.allclose(x0[:, 1:, 0, 1], torch.tensor([[0.1, -0.05, 0.3]]))   # r = 1: the prior
    assert torch.allclose(x0[:, 1:, 1, 0], torch.tensor([[0.55, -0.025, 0.15]]))


def test_inputs_rgb_four_maps_and_t():
    net = grat.GRATConvNet.__new__(grat.GRATConvNet)
    xc, xt = torch.rand(2, 3, 4, 4), torch.rand(2, 4, 4, 4)
    t = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1)
    z = net.inputs(xc, xt, t)
    assert z.shape == (2, grat.CHANNELS, 4, 4)
    assert torch.equal(z[:, :3], xc) and torch.equal(z[:, 3:7], xt) and z[1, 7, 0, 0] == 0.75


def test_flow_losses_per_channel():
    v, g = torch.zeros(2, 4, 3, 3), torch.ones(2, 4, 3, 3)
    g[:, 2] = 2.0
    assert torch.allclose(grat.flow_losses(v, g), torch.tensor([1.0, 1.0, 4.0, 1.0]))


def test_released_checkpoint_options_become_a_config():
    net = grat.GRATConvNet.__new__(grat.GRATConvNet)
    net.__setstate__({"p": {"pc": 0.1, "ps": -0.05, "pw": 0.3, "src": "rel", "ot": 1, "lam": 20.0}})
    assert net.cfg == grat.Config(lam=20.0, pose_prior=(0.1, -0.05, 0.3))
    net = grat.GRATGGCNN.__new__(grat.GRATGGCNN)
    net.__setstate__({"p": {"pc": 0.1, "ps": -0.05, "pw": 0.3}})
    assert net.cfg == grat.Config(source="constant", pose_prior=(0.1, -0.05, 0.3)) and not net.cfg.uses_ot


def test_released_module_names_resolve():
    grat.accept_released_checkpoints()
    import sys
    assert sys.modules["lgdbench.ideas.r19_flowgeo"].R19GG is grat.GRATGGCNN


def test_lambda_of_each_pipeline():
    assert set(grat.LAMBDA) == {"lgrconvnet3", "lggcnn", "crog"} and all(v > 0 for v in grat.LAMBDA.values())
