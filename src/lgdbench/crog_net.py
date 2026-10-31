"""CROG (Tziafas et al., CoRL 2023) behind the LGD model interface.

The official ``model.crog.CROG`` runs unchanged; ``CROGGrasp`` only adapts its inputs and outputs to the
``compute_loss(xc, yc, prompt, query)`` interface that the official LGD ``train()``, ``validate()`` and ``evaluate.py``
call, so CROG is trained and scored by exactly the same loop and metric as the other baselines.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import refs

CROG = refs.REF_ROOT / "CROG"
CROG_COMMIT = "1eeee85de1fe6bffdc66c9ed9a622028ea04578e"
CROG_CONFIG = CROG / "config" / "OCID-VLG" / "crog_multiple_r50.yaml"


def _is_utils(name: str) -> bool:
    return name == "utils" or name.startswith("utils.")


@contextmanager
def _crog_on_path():
    """CROG and LGD both ship a top-level ``utils`` package. While CROG modules import, ``utils`` resolves to CROG's;
    afterwards LGD's ``utils`` modules are put back. Modules CROG imported keep their own references."""
    saved = {k: sys.modules.pop(k) for k in [k for k in sys.modules if _is_utils(k)]}
    sys.path.insert(0, str(CROG))
    try:
        yield
    finally:
        sys.path.remove(str(CROG))
        for k in [k for k in sys.modules if _is_utils(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


@lru_cache(maxsize=None)
def import_crog() -> types.SimpleNamespace:
    """The official CROG pieces used here. Leaves the CROG ``model`` package in ``sys.modules``, which is what
    ``torch.load`` needs to unpickle a saved ``CROGGrasp``."""
    refs.require(CROG)
    with _crog_on_path():
        import model as crog_model
        from utils.config import load_cfg_from_cfg_file
        from utils.dataset import tokenize
    return types.SimpleNamespace(build_crog=crog_model.build_crog, tokenize=tokenize, load_cfg=load_cfg_from_cfg_file)


@lru_cache(maxsize=None)
def word_len() -> int:
    """CROG's text length (``crog_multiple_r50.yaml``: 17), for tokenising outside the model."""
    return int(import_crog().load_cfg(str(CROG_CONFIG)).word_len)


def load_config(clip_pretrain: str):
    """``crog_multiple_r50.yaml`` (the OCID-VLG multiple-object setting) with the local path of RN50.pt and plain BN
    (one GPU, no DDP)."""
    cfg = import_crog().load_cfg(str(CROG_CONFIG))
    cfg.clip_pretrain = clip_pretrain
    cfg.sync_bn = False
    return cfg


class CROGGrasp(nn.Module):
    """``model.crog.CROG`` with LGD's ``compute_loss`` interface.

    Inputs: ``xc`` the CLIP-normalised 224x224 image (``data.make_crog_dataset_class``), ``query`` the text prompts (GA++ grasp instructions).
    Targets ``yc`` = (pos, cos, sin, width[, object mask]) from the LGD dataset; the mask is only present in training.
    Outputs: CROG's quality/cos/sin/width maps, bilinearly resized to the input size, as ``pred`` = pos/cos/sin/width.
    """

    def __init__(self, crog: nn.Module, word_len: int):
        super().__init__()
        self.crog = crog
        self.word_len = word_len

    def tokens(self, query, device):
        return import_crog().tokenize(list(query), self.word_len, True).to(device)

    def compute_loss(self, xc, yc, prompt, query):
        y_pos, y_cos, y_sin, y_width = yc[:4]
        words = self.tokens(query, xc.device)
        if self.training:
            if bool((yc[4] < 0).any()):
                raise ValueError("training batch holds a sample without object mask (data.make_crog_dataset_class)")
            # CROG.forward computes its own loss in training mode (model/crog.py:76-111); argument order: mask, qua, sin, cos, wid
            preds, _, loss, loss_dict = self.crog(xc, words, yc[4], y_pos, y_sin, y_cos, y_width)
            losses = {k: torch.as_tensor(float(v)) for k, v in loss_dict.items()}
        else:
            preds, _ = self.crog(xc, words)
        _, qua, sin, cos, wid = preds
        size = xc.shape[-2:]
        pred = {k: F.interpolate(v, size=size, mode="bilinear", align_corners=False)
                for k, v in (("pos", qua), ("cos", cos), ("sin", sin), ("width", wid))}
        if not self.training:
            # validation loss for logging only, CROG's grasp terms (smooth L1) on the resized maps
            losses = {f"m_{k}": F.smooth_l1_loss(pred[p], t)
                      for k, p, t in (("qua", "pos", y_pos), ("sin", "sin", y_sin), ("cos", "cos", y_cos),
                                      ("wid", "width", y_width))}
            loss = sum(losses.values())
        return {"loss": loss, "losses": losses, "pred": pred}


def build(clip_pretrain: str):
    """``(CROGGrasp, param_list)``: the official ``build_crog`` (backbone/head parameter groups, model/__init__.py)."""
    cfg = load_config(clip_pretrain)
    crog, param_list = import_crog().build_crog(cfg)
    return CROGGrasp(crog, cfg.word_len), param_list, cfg
