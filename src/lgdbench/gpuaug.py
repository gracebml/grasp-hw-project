from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import CLIP_MEAN, CLIP_STD

SRC = 416  # Grasp-Anything image size


def transform(img: torch.Tensor, rot: float, zoom: float, out_size: int = 224) -> torch.Tensor:
    """C x 416 x 416 -> C x out x out, as the official ``get_rgb``."""
    k = int(round(float(rot) / (math.pi / 2))) % 4
    img = torch.rot90(img, k, dims=(1, 2))
    sr = int(SRC * (1 - float(zoom))) // 2
    if sr:
        img = img[:, sr:SRC - sr, sr:SRC - sr]
    return F.interpolate(img[None], size=(out_size, out_size), mode="bilinear", align_corners=False,
                         antialias=True)[0]


def normalise(x: torch.Tensor, norm: str) -> torch.Tensor:
    """``x``: B x 3 x H x W in [0, 1]. ``lgd``: minus the image mean; ``clip``: CLIP mean and std."""
    if norm == "lgd":
        return x - x.mean(dim=(1, 2, 3), keepdim=True)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def transform_masks(m: torch.Tensor, rot, zoom, out_size: int) -> torch.Tensor:
    """Masks get the image transform and a 0.5 threshold; absent masks (all -1) stay -1."""
    missing = (m < 0).flatten(2).all(dim=2)
    m = torch.stack([transform(m[i], rot[i], zoom[i], out_size) for i in range(m.shape[0])])
    m = (m > 0.5).float()
    m[missing] = -1.0
    return m


class GPUAugLoader:
    def __init__(self, loader, device, norm: str = "lgd", out_size: int = 224):
        self.loader, self.device, self.norm, self.out_size = loader, device, norm, out_size

    @property
    def dataset(self):
        return self.loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y, didx, rot, zoom, prompt, query in self.loader:
            x = x.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            x = torch.stack([transform(x[i], rot[i], zoom[i], self.out_size) for i in range(x.shape[0])])
            y = [t.to(self.device, non_blocking=True) for t in y]
            if self.norm == "clip" and len(y) >= 5:  # CROG's object mask
                y[4] = transform_masks(y[4], rot, zoom, self.out_size)
            yield normalise(x, self.norm), y, didx, rot, zoom, prompt, query
