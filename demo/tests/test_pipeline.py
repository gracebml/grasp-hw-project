import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as P  # noqa: E402

EXAMPLE = P.REPO / "visualization"
QUERY = "Lift spoon by its handle."
GG_GRAT = P.CKPT_ROOT / "grat" / "lggcnn_s123" / "epoch_11_iou_0.42"
needs_ckpt = pytest.mark.skipif(not GG_GRAT.exists(), reason="released checkpoint not under checkpoints/")


def test_square_crop_keeps_centre():
    a = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert np.array_equal(P.square_crop(a), a[:, 1:5])
    assert P.square_crop(a.transpose(1, 0, 2)).shape == (4, 4, 3)


def test_preprocess_normalisations():
    rgb = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    x, shown = P.preprocess(rgb, crog=False)
    assert x.shape == (1, 3, 224, 224) and shown.shape == (224, 224, 3) and shown.dtype == np.uint8
    assert abs(float(x.mean())) < 1e-5                 # LGD: /255 minus one scalar mean
    assert torch.allclose(x[0].permute(1, 2, 0), torch.from_numpy(shown / 255.0).float() - float(shown.mean() / 255.0),
                          atol=2e-3)
    xc, _ = P.preprocess(rgb, crog=True)
    std, mean = torch.tensor(P.CLIP_STD).view(3, 1, 1), torch.tensor(P.CLIP_MEAN).view(3, 1, 1)
    assert torch.allclose(xc[0] * std + mean, torch.from_numpy(shown / 255.0).permute(2, 0, 1).float(), atol=2e-3)


def test_list_checkpoints(tmp_path):
    for rel in ["grat/lggcnn_s123/epoch_05_iou_0.30", "grat/lggcnn_s123/epoch_11_iou_0.42",
                "grat/lggcnn_s123/epoch_11_iou_0.42.sha256", "baseline/crog_s456/epoch_29_iou_0.45",
                "grat_no_ot/lggcnn_s123/epoch_00_iou_0.10", "grat/notes/epoch_01_iou_0.99"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    got = {(c.group, c.network, c.seed): c.path.name for c in P.list_checkpoints(tmp_path)}
    assert got == {("grat", "lggcnn", 123): "epoch_11_iou_0.42", ("baseline", "crog", 456): "epoch_29_iou_0.45"}


@needs_ckpt
def test_reproduces_paper_example():
    """With the PIL resize of visualization/run_example.py, the demo gives the maps of visualization/example/maps.npz."""
    ref = np.load(EXAMPLE / "example" / "maps.npz")
    rgb = np.asarray(Image.open(EXAMPLE / "data" / "example_image.png").convert("RGB").resize((224, 224), Image.BICUBIC))
    res = P.run(P.load_model(GG_GRAT), rgb, QUERY)
    assert np.abs(res.relevance - ref["rel"]).max() < 1e-4
    assert np.abs(res.q - ref["q"]).max() < 1e-4
    assert np.abs(res.width - ref["w"]).max() < 1e-2
    assert res.grasps[0][:2] == tuple(int(v) for v in ref["grasp"][:2])
    assert len(res.states) == 9                        # x_0 and the 8 Euler steps


@needs_ckpt
def test_unrolled_steps_end_at_compute_loss():
    net = P.load_model(GG_GRAT)
    rgb = np.asarray(Image.open(EXAMPLE / "data" / "example_image.png").convert("RGB"))
    res = P.run(net, rgb, QUERY)
    xc, _ = P.preprocess(rgb, crog=False)
    with torch.no_grad():
        pred = net.compute_loss(xc, [torch.zeros(1, 1, 224, 224)] * 4, [QUERY], [QUERY])["pred"]
    assert np.array_equal(res.states[-1]["q"], pred["pos"][0, 0].numpy())
    assert np.array_equal(res.states[-1]["width"], pred["width"][0, 0].numpy() * 150.0)


def _maps_with_peak(peak: float) -> torch.Tensor:
    yy, xx = np.mgrid[:224, :224]
    q = peak * np.exp(-((yy - 100) ** 2 + (xx - 120) ** 2) / (2 * 8.0 ** 2))
    one = np.ones_like(q)
    return torch.from_numpy(np.stack([q, one, 0 * one, 0.3 * one])[None]).float()   # q, cos, sin, width / 150


def test_no_grasp_below_threshold_gives_only_a_candidate():
    res = P.Result(image=np.zeros((224, 224, 3), np.uint8), text="", ckpt="", is_grat=False)
    P._decode(res, _maps_with_peak(0.15), n_grasps=3)
    assert res.grasps == [] and res.best is not None
    assert res.best[:2] == (100, 120) and res.best[5] < P.THRESHOLD
    assert abs(res.best[3] - 45.0) < 1e-3 and abs(res.best[4] - 22.5) < 1e-3     # width 0.3 x 150, jaw = half


def test_grasp_above_threshold_has_no_candidate():
    res = P.Result(image=np.zeros((224, 224, 3), np.uint8), text="", ckpt="", is_grat=False)
    P._decode(res, _maps_with_peak(0.6), n_grasps=3)
    assert [g[:2] for g in res.grasps] == [(100, 120)] and res.best is None
