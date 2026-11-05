"""Inference of the released checkpoints for the demo, without Streamlit.

Everything model-side is called from the repository (``lgdbench``) and the official LGD / CROG code; this file only
loads images, runs the networks step by step and keeps what is needed to draw the steps.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lgdbench import crog_net, grat, refs  # noqa: E402
from lgdbench.data import CLIP_MEAN, CLIP_STD  # noqa: E402
from lgdbench.relevance import rgb_from_lgd  # noqa: E402

refs.add_lgd_to_path()
from inference.post_process import post_process_output  # noqa: E402
from utils.dataset_processing import image as lgd_image  # noqa: E402
from utils.dataset_processing.grasp import Grasp, detect_grasps  # noqa: E402

CKPT_ROOT = Path(os.environ.get("GRAT_CKPT_ROOT", REPO / "checkpoints"))
GROUPS = {"baseline": "Baseline", "grat": "+ GRAT"}
NETWORKS = {"lgrconvnet3": "GR-ConvNet + CLIP", "lggcnn": "GG-CNN + CLIP", "crog": "CROG"}
SIZE = 224                      # protocol: 416 x 416 resized to 224
THRESHOLD = 0.2                 # threshold_abs of the official detect_grasps
CKPT_NAME = re.compile(r"epoch_(\d+)_iou_([0-9.]+)$")
RUN_DIR = re.compile(r"^(lgrconvnet3|lggcnn|crog)_s(\d+)$")


@dataclass(frozen=True)
class Checkpoint:
    group: str
    network: str
    seed: int
    path: Path

    @property
    def label(self) -> str:
        return f"{NETWORKS[self.network]} {GROUPS[self.group]}, seed {self.seed}"


def list_checkpoints(root: Path = CKPT_ROOT) -> list[Checkpoint]:
    """One checkpoint per run folder ``<root>/<group>/<network>_s<seed>/epoch_XX_iou_Y`` (groups baseline and grat).
    The released folders hold the best-validation epoch only; with several, the highest val IoU of the file name."""
    out = []
    for group in GROUPS:
        for run in sorted((root / group).glob("*_s*")):
            m = RUN_DIR.match(run.name)
            files = [(float(c[2]), p) for p in run.iterdir() if (c := CKPT_NAME.search(p.name))] if m else []
            if files:
                out.append(Checkpoint(group, m[1], int(m[2]), max(files)[1]))
    return out


def load_model(path: str | os.PathLike) -> torch.nn.Module:
    grat.accept_released_checkpoints()
    crog_net.import_crog()      # lets torch.load unpickle CROG
    net = torch.load(str(path), map_location="cpu", weights_only=False)
    return net.float().eval()


def square_crop(rgb: np.ndarray) -> np.ndarray:
    """Centre square of an H x W x 3 uint8 image (demo only: Grasp-Anything images are already square)."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return rgb[top:top + s, left:left + s]


def preprocess(rgb: np.ndarray, crog: bool) -> tuple[torch.Tensor, np.ndarray]:
    """(network input 1 x 3 x 224 x 224, the 224 x 224 uint8 image shown). Resize of the official ``get_rgb``; then
    LGD's ``normalise`` (/255, minus the image mean) or, for CROG, the CLIP normalisation of ``data.py``."""
    img = lgd_image.Image(square_crop(rgb[..., :3]))
    img.resize((SIZE, SIZE))
    shown = np.clip(img.img, 0, 255).astype(np.uint8)
    if crog:
        x = (img.img.astype(np.float32) / 255.0 - np.asarray(CLIP_MEAN, np.float32)) / np.asarray(CLIP_STD, np.float32)
    else:
        img.normalise()
        x = img.img
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float(), shown


def maps_of(x: torch.Tensor) -> dict[str, np.ndarray]:
    """Raw 4-channel state (q, cos 2θ, sin 2θ, width/150) -> displayable maps, before LGD's smoothing."""
    q, c, s, w = x[0].numpy()
    return {"q": q, "angle": np.arctan2(s, c) / 2, "width": w * 150.0}


@dataclass
class Result:
    image: np.ndarray                      # 224 x 224 x 3 uint8
    text: str
    ckpt: str
    is_grat: bool
    relevance: np.ndarray | None = None    # r, 224 x 224 in [0, 1]
    states: list = field(default_factory=list)   # GRAT: x_0 ... x_K as dicts of maps (maps_of)
    q: np.ndarray | None = None            # after LGD's post_process_output
    angle: np.ndarray | None = None
    width: np.ndarray | None = None
    grasps: list = field(default_factory=list)   # [(row, col, angle, length, width, quality, corners 4 x 2 (row, col))]
    best: tuple | None = None              # no grasp passed THRESHOLD: the same tuple at the highest quality, for display
    timings: dict = field(default_factory=dict)


def _as_tuple(g: Grasp, q: np.ndarray) -> tuple:
    r, c = g.center
    return (int(r), int(c), float(g.angle), float(g.length), float(g.width), float(q[r, c]),
            np.asarray(g.as_gr.points, dtype=float))


def _decode(res: Result, pred: torch.Tensor, n_grasps: int) -> None:
    q, ang, w = post_process_output(*pred.split(1, 1))
    res.q, res.angle, res.width = q, ang, w
    res.grasps = [_as_tuple(g, q) for g in detect_grasps(q, ang, width_img=w, no_grasps=n_grasps)]
    if not res.grasps:
        # what detect_grasps would build at the highest quality without its threshold; not a detection
        r, c = np.unravel_index(int(np.argmax(q)), q.shape)
        g = Grasp((r, c), ang[r, c])
        g.length = w[r, c]
        g.width = g.length / 2
        res.best = _as_tuple(g, q)


@torch.no_grad()
def run(net: torch.nn.Module, rgb: np.ndarray, text: str, n_grasps: int = 1, ckpt: str = "") -> Result:
    crog = isinstance(net, crog_net.CROGGrasp)
    is_grat = isinstance(net, grat._Flow)
    xc, shown = preprocess(rgb, crog)
    res = Result(image=shown, text=text, ckpt=ckpt, is_grat=is_grat)
    query = [text]
    t0 = time.perf_counter()
    if not is_grat:
        dummy = [torch.zeros(1, 1, SIZE, SIZE) for _ in range(4)]
        p = net.compute_loss(xc, dummy, query, query)["pred"]
        pred = torch.cat([p["pos"], p["cos"], p["sin"], p["width"]], 1)
        res.timings["forward"] = time.perf_counter() - t0
    else:
        # the inference branch of GRAT's compute_loss (grat.py), unrolled to keep every Euler state
        cfg = net.cfg
        if crog:
            rel = net.relevance(xc * net.relevance.std + net.relevance.mean, query)
            words = net.tokens(query, xc.device)
        else:
            rel = net.relevance(rgb_from_lgd(xc), query)
        res.timings["relevance r (CLIP)"] = time.perf_counter() - t0
        res.relevance = rel[0, 0].numpy()
        x = grat.source_maps(rel, cfg, sample=False)
        res.states.append(maps_of(x))
        t1 = time.perf_counter()
        for k in range(cfg.steps):
            t = grat.time_grid(k, cfg.steps, xc)
            if crog:
                (_, vq, vs, vc, vw), _ = net.run(xc, words, x, t)
                v = F.interpolate(torch.cat([vq, vc, vs, vw], 1), size=xc.shape[-2:], mode="bilinear",
                                  align_corners=False)
            else:
                v = torch.cat(net(net.inputs(xc, x, t), query, query), 1)
            x = x + v / cfg.steps
            res.states.append(maps_of(x))
        res.timings[f"{cfg.steps} Euler steps"] = time.perf_counter() - t1
        pred = x
    t2 = time.perf_counter()
    _decode(res, pred, n_grasps)
    res.timings["decode (LGD)"] = time.perf_counter() - t2
    return res
