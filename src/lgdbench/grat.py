"""GRAT (Grounding as Transport): a rectified flow carries all four grasp maps (quality, cos 2theta, sin 2theta, width)
from the instruction's CLIP relevance map r to the target.

Source and path: x0 = [r, r p + (1 - r) b] (p: mean ground-truth pose on grasp pixels, b: label value outside grasps),
x_t = (1 - t) x0 + t x1, plus Gaussian noise on x0 in training. The network regresses x1 - x0.
Loss: mean over the channels of the squared velocity error, plus lambda times the OT terms (``lgdbench.ot``) on the one-step
endpoint x_t + (1 - t) v. CROG keeps its mask loss.
Inference: Euler steps from the noise-free x0; the four maps go to the pipeline's official decoder. Evaluation runs inside
``compute_loss`` (eval mode), so the official ``evaluate.py`` and validation call it unchanged.

GR-ConvNet and GG-CNN take (x_t, t) as extra input channels and their four heads predict the velocity. CROG's backbone takes
exactly three channels, so an encoder of (x_t, t) is added to the four grasp slices of its projector.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ot, refs

refs.add_lgd_to_path()

from .crog_net import CROGGrasp  # noqa: E402
from .fixed_models import GenerativeResnetNoDetach, LGGCNNNoDetach  # noqa: E402
from .relevance import ClipRelevance, rgb_from_lgd  # noqa: E402

BACKGROUND = (1.0, 0.0, 0.0)   # GT cos 2theta, sin 2theta, width outside the grasp rectangles (grasp_data.py:74-93)
POSE_PRIOR = (0.013, -0.0001, 0.8804)   # mean GT (cos, sin, width) on grasp pixels, experiments/calibrate.py
LAMBDA = {"lgrconvnet3": 20.0, "lggcnn": 305.0, "crog": 7.5}   # 0.5 |g_flow| / |g_OT| at initialisation, experiments/calibrate.py
CHANNELS = 3 + 4 + 1   # RGB, x_t, t


@dataclass(frozen=True)
class Config:
    lam: float = 0.0                       # weight of the OT loss; 0 = no OT loss (the ablation)
    source: str = "relevance"              # pose channels of x0: "relevance" (the paper), "constant" (the ablation)
    pose_prior: tuple = POSE_PRIOR
    steps: int = 8
    sigma: float = 0.1
    tau: float = 0.1
    margin: float = 0.05
    eps: float = 0.05
    iters: int = 50

    def __post_init__(self):
        if self.source not in ("relevance", "constant"):
            raise ValueError(f"source must be relevance or constant, got {self.source!r}")
        if self.lam < 0 or self.steps < 1:
            raise ValueError(f"bad config {self}")

    @property
    def uses_ot(self) -> bool:
        return self.lam > 0


def source_maps(rel: torch.Tensor, cfg: Config, sample: bool) -> torch.Tensor:
    """x0 as B x 4 x H x W: the relevance map, then the pose channels; + sigma noise on every channel when sampling."""
    pose = torch.tensor(cfg.pose_prior, dtype=rel.dtype, device=rel.device).view(1, 3, 1, 1)
    if cfg.source == "relevance":
        bg = torch.tensor(BACKGROUND, dtype=rel.dtype, device=rel.device).view(1, 3, 1, 1)
        pose_maps = rel * pose + (1 - rel) * bg
    else:
        pose_maps = pose.expand(rel.shape[0], 3, *rel.shape[-2:])
    x0 = torch.cat([rel, pose_maps], 1)
    return x0 + cfg.sigma * torch.randn_like(x0) if sample else x0


def flow_losses(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Mean squared velocity error of each of the four channels."""
    se = (v - g) ** 2
    return torch.stack([se[:, c].mean() for c in range(4)])


def time_grid(k: int, steps: int, xc: torch.Tensor) -> torch.Tensor:
    return torch.full((xc.shape[0], 1, 1, 1), k / steps, device=xc.device)


class _Flow:
    """What the three pipelines share: the frozen relevance model."""

    def setup_flow(self, cfg: Config):
        self.cfg = cfg
        self.relevance = ClipRelevance()

    def train(self, mode: bool = True):
        super().train(mode)
        self.relevance.model.eval()  # frozen, like the pipelines' text encoders
        return self

    def __setstate__(self, state):
        super().__setstate__(state)
        if "cfg" not in self.__dict__:         # released checkpoints: their options sit in a dict ``p``
            p = self.p
            self.cfg = Config(lam=p["lam"] if p.get("ot") else 0.0, source="relevance" if p.get("src") == "rel" else "constant",
                              pose_prior=(p["pc"], p["ps"], p["pw"]))


class _DensePipeline(_Flow):
    """GR-ConvNet and GG-CNN: (x_t, t) enter as input channels, the four heads become the velocity."""

    def inputs(self, xc, xt, t):
        return torch.cat([xc, xt, t.expand(-1, 1, *xt.shape[-2:])], 1)

    def compute_loss(self, xc, yc, prompt, query):
        y = torch.cat(list(yc[:4]), 1)        # pos, cos, sin, width
        rel = self.relevance(rgb_from_lgd(xc), query)
        extra = {}
        if self.training:
            x0 = source_maps(rel, self.cfg, sample=True)
            t = torch.rand(xc.shape[0], 1, 1, 1, device=xc.device)
            xt = (1 - t) * x0 + t * y
            v = torch.cat(self(self.inputs(xc, xt, t), prompt, query), 1)
            per = flow_losses(v, y - x0)
            loss = per.mean()
            x = xt + (1 - t) * v              # one-step endpoint
            if self.cfg.uses_ot:
                c = self.cfg
                extra["ot_pull"], extra["ot_margin"] = ot.transport_terms(x[:, :1], yc[0], yc[4], c.tau, c.margin, c.eps, c.iters)
                loss = loss + c.lam * (extra["ot_pull"] + extra["ot_margin"])
        else:
            x = source_maps(rel, self.cfg, sample=False)
            for k in range(self.cfg.steps):
                t = time_grid(k, self.cfg.steps, xc)
                x = x + torch.cat(self(self.inputs(xc, x, t), prompt, query), 1) / self.cfg.steps
            per = torch.stack([F.smooth_l1_loss(x[:, i:i + 1], y[:, i:i + 1]) for i in range(4)])
            loss = per.sum()
        pos, cos, sin, width = x.split(1, 1)
        return {"loss": loss,
                "losses": {"p_loss": per[0], "cos_loss": per[1], "sin_loss": per[2], "width_loss": per[3], **extra},
                "pred": {"pos": pos, "cos": cos, "sin": sin, "width": width}}


class GRATConvNet(_DensePipeline, GenerativeResnetNoDetach):
    """GRAT on GR-ConvNet + CLIP."""

    def __init__(self, cfg: Config):
        GenerativeResnetNoDetach.__init__(self, input_channels=CHANNELS, dropout=1, prob=0.1, channel_size=32)
        self.setup_flow(cfg)


class GRATGGCNN(_DensePipeline, LGGCNNNoDetach):
    """GRAT on GG-CNN + CLIP."""

    def __init__(self, cfg: Config):
        LGGCNNNoDetach.__init__(self, input_channels=CHANNELS, dropout=1, prob=0.1, channel_size=32)
        self.setup_flow(cfg)


class GRATCROG(_Flow, CROGGrasp):
    """GRAT on CROG: an encoder of (x_t, t) is added to the qua, sin, cos and wid slices of ``proj.vis`` (slice order mask, qua,
    sin, cos, wid; ``model/layers.py:69-105``). The hook is a method of this module, so the network stays picklable."""

    def __init__(self, crog, word_len: int, cfg: Config):
        CROGGrasp.__init__(self, crog, word_len)
        self.setup_flow(cfg)
        c = crog.proj.in_dim
        self.flow_enc = nn.Sequential(nn.Conv2d(5, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 4 * c, 1))
        self._xt = self._t = None
        crog.proj.vis.register_forward_hook(self._add_xt)

    def _add_xt(self, module, inputs, output):
        xt, t = self._xt, self._t
        self._xt = self._t = None
        if xt is None:
            return None
        c = self.crog.proj.in_dim
        xt = F.interpolate(xt.to(output.dtype), output.shape[-2:], mode="bilinear", align_corners=False)
        e = self.flow_enc(torch.cat([xt, t.to(output.dtype).expand(-1, 1, *xt.shape[-2:])], 1))
        return torch.cat([output[:, :c], output[:, c:] + e], 1)

    def run(self, xc, words, xt, t, targets=None):
        self._xt, self._t = xt, t
        return self.crog(xc, words, *targets) if targets is not None else self.crog(xc, words)

    def compute_loss(self, xc, yc, prompt, query):
        y = torch.cat(list(yc[:4]), 1)        # pos, cos, sin, width
        words = self.tokens(query, xc.device)
        rel = self.relevance(xc * self.relevance.std + self.relevance.mean, query)
        size = xc.shape[-2:]

        def up(m):
            return F.interpolate(m, size=size, mode="bilinear", align_corners=False)

        if self.training:
            if bool((yc[4] < 0).any()):
                raise ValueError("training batch holds a sample without object mask (data.make_crog_dataset_class)")
            x0 = source_maps(rel, self.cfg, sample=True)
            t = torch.rand(xc.shape[0], 1, 1, 1, device=xc.device)
            xt = (1 - t) * x0 + t * y
            g = y - x0                        # velocity targets: pos, cos, sin, width
            # CROG's mask BCE is kept; its four smooth-L1 grasp terms are replaced by the flow loss. CROG detaches its outputs
            # (model/crog.py:76-111), so the live maps come from a temporary hook on ``crog.proj``.
            live = {}
            handle = self.crog.proj.register_forward_hook(lambda m, i, o: live.__setitem__("maps", o))
            try:  # argument order of CROG.forward: mask, qua, sin, cos, wid
                _, tg, total, loss_dict = self.run(xc, words, xt, t, (yc[4], g[:, 0:1], g[:, 2:3], g[:, 1:2], g[:, 3:4]))
            finally:
                handle.remove()
            _, vq, vs, vc, vw = live["maps"]
            _, tq, ts, tcos, tw = tg          # targets at the prediction size (CROG's nearest resize)
            crog_grasp = sum(F.smooth_l1_loss(a, b) for a, b in ((vq, tq), (vs, ts), (vc, tcos), (vw, tw)))
            per = flow_losses(torch.cat([vq, vc, vs, vw], 1), torch.cat([tq, tcos, ts, tw], 1))
            loss = total - crog_grasp + per.mean()
            losses = {"m_ins": torch.as_tensor(float(loss_dict["m_ins"])), "v_pos": per[0].detach(),
                      "v_cos": per[1].detach(), "v_sin": per[2].detach(), "v_wid": per[3].detach()}
            x = xt + (1 - t) * up(torch.cat([vq, vc, vs, vw], 1))
            if self.cfg.uses_ot:              # mask (5th target), then the other object's quality map
                c = self.cfg
                pull, push = ot.transport_terms(x[:, :1], yc[0], yc[5], c.tau, c.margin, c.eps, c.iters)
                losses["ot_pull"], losses["ot_margin"] = pull.detach(), push.detach()
                loss = loss + c.lam * (pull + push)
            x = x.detach()
        else:
            x = source_maps(rel, self.cfg, sample=False)
            for k in range(self.cfg.steps):
                (_, vq, vs, vc, vw), _ = self.run(xc, words, x, time_grid(k, self.cfg.steps, xc))
                x = x + up(torch.cat([vq, vc, vs, vw], 1)) / self.cfg.steps
        pos, cos, sin, width = x.split(1, 1)
        pred = {"pos": pos, "cos": cos, "sin": sin, "width": width}
        if not self.training:
            losses = {f"m_{k}": F.smooth_l1_loss(pred[p], yc[i])
                      for i, (k, p) in enumerate((("qua", "pos"), ("cos", "cos"), ("sin", "sin"), ("wid", "width")))}
            loss = sum(losses.values())
        return {"loss": loss, "losses": losses, "pred": pred}


def accept_released_checkpoints() -> None:
    """The released checkpoints pickle the module names of the development tree; map them to the classes above. Call before
    ``torch.load``."""
    package = types.ModuleType("lgdbench.ideas")
    package.__path__ = []
    flow, rel = types.ModuleType("lgdbench.ideas.r19_flowgeo"), types.ModuleType("lgdbench.ideas.r01_promptflow")
    flow.R19Net, flow.R19GG, flow.R19CROG = GRATConvNet, GRATGGCNN, GRATCROG
    rel.ClipRelevance = ClipRelevance
    sys.modules.update({package.__name__: package, flow.__name__: flow, rel.__name__: rel})


def build(network: str, cfg: Config, clip_rn50: str | None = None):
    """(net, parameters for the optimizer, CROG's config or None) for ``network`` = lgrconvnet3 | lggcnn | crog."""
    if network == "crog":
        from . import crog_net
        crog_cfg = crog_net.load_config(clip_rn50)
        crog, param_list = crog_net.import_crog().build_crog(crog_cfg)
        net = GRATCROG(crog, crog_cfg.word_len, cfg)
        # the encoder trains with the projector's (non-backbone) learning rate
        param_list[-1]["params"] = list(param_list[-1]["params"]) + list(net.flow_enc.parameters())
        return net, param_list, crog_cfg
    classes = {"lgrconvnet3": GRATConvNet, "lggcnn": GRATGGCNN}
    if network not in classes:
        raise ValueError(f"unknown network {network!r}")
    net = classes[network](cfg)
    return net, [q for q in net.parameters() if q.requires_grad], None
