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
    """CROG and LGD both ship a top-level ``utils``: import CROG with its own, then put LGD's back."""
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
    """The official CROG pieces used here; also lets ``torch.load`` unpickle a saved ``CROGGrasp``."""
    refs.require(CROG)
    with _crog_on_path():
        import model as crog_model
        from utils.config import load_cfg_from_cfg_file
        from utils.dataset import tokenize
    return types.SimpleNamespace(build_crog=crog_model.build_crog, tokenize=tokenize, load_cfg=load_cfg_from_cfg_file)


@lru_cache(maxsize=None)
def word_len() -> int:
    """CROG's text length (17)."""
    return int(import_crog().load_cfg(str(CROG_CONFIG)).word_len)


def load_config(clip_pretrain: str):
    """The official config, with the local RN50.pt and plain batch norm (one GPU)."""
    cfg = import_crog().load_cfg(str(CROG_CONFIG))
    cfg.clip_pretrain = clip_pretrain
    cfg.sync_bn = False
    return cfg


class CROGGrasp(nn.Module):
    """CROG with LGD's interface. Targets: (pos, cos, sin, width[, object mask in training]); maps resized to the input."""

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
            # CROG computes its own loss in training mode; argument order: mask, qua, sin, cos, wid
            preds, _, loss, loss_dict = self.crog(xc, words, yc[4], y_pos, y_sin, y_cos, y_width)
            losses = {k: torch.as_tensor(float(v)) for k, v in loss_dict.items()}
        else:
            preds, _ = self.crog(xc, words)
        _, qua, sin, cos, wid = preds
        size = xc.shape[-2:]
        pred = {k: F.interpolate(v, size=size, mode="bilinear", align_corners=False)
                for k, v in (("pos", qua), ("cos", cos), ("sin", sin), ("width", wid))}
        if not self.training:
            # for logging only
            losses = {f"m_{k}": F.smooth_l1_loss(pred[p], t)
                      for k, p, t in (("qua", "pos", y_pos), ("sin", "sin", y_sin), ("cos", "cos", y_cos),
                                      ("wid", "width", y_width))}
            loss = sum(losses.values())
        return {"loss": loss, "losses": losses, "pred": pred}


def build(clip_pretrain: str):
    """(network, CROG's parameter groups, config)."""
    cfg = load_config(clip_pretrain)
    crog, param_list = import_crog().build_crog(cfg)
    return CROGGrasp(crog, cfg.word_len), param_list, cfg
