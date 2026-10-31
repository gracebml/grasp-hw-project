"""Image augmentation on the GPU.

The official loader rotates, zooms and resizes every 416x416 image with skimage on the CPU (~95 ms per sample), which
leaves the GPU mostly idle. Here the official ``__getitem__`` still draws ``rot``/``zoom`` and builds the targets
(``data.make_gpuaug_dataset_class``); ``GPUAugLoader`` applies the same image operations, in the same order, on the GPU:

    Image.rotate(rot)  -> torch.rot90, k = rot / (pi/2), counter-clockwise like skimage.transform.rotate
    Image.zoom(zoom)   -> centre crop with sr = int(416 * (1 - zoom)) // 2 (utils/dataset_processing/image.py)
    Image.resize(224)  -> bilinear with anti-aliasing (the official resize back to 416 and then to 224 in one step)
    Image.normalise()  -> /255 minus the image mean (baselines) or CLIP mean/std (CROG)

The wrapped loader yields batches already on the GPU in the official (x, y, didx, rot, zoom, prompt, query) layout, so the
official ``train()`` / ``validate()`` run unchanged; ``.dataset`` is the underlying dataset (``validate`` calls ``get_gtbb``).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import CLIP_MEAN, CLIP_STD

SRC = 416  # Grasp-Anything image size (grasp_anywhere_data.py)


def transform(img: torch.Tensor, rot: float, zoom: float, out_size: int = 224) -> torch.Tensor:
    """``img``: C x 416 x 416 float on the GPU -> C x out x out, as ``get_rgb`` (rotate, zoom, resize)."""
    k = int(round(float(rot) / (math.pi / 2))) % 4
    img = torch.rot90(img, k, dims=(1, 2))
    sr = int(SRC * (1 - float(zoom))) // 2
    if sr:
        img = img[:, sr:SRC - sr, sr:SRC - sr]
    return F.interpolate(img[None], size=(out_size, out_size), mode="bilinear", align_corners=False,
                         antialias=True)[0]


def normalise(x: torch.Tensor, norm: str) -> torch.Tensor:
    """``x``: B x 3 x H x W in [0, 1]."""
    if norm == "lgd":  # Image.normalise: /255 then minus the mean of the whole image
        return x - x.mean(dim=(1, 2, 3), keepdim=True)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def transform_masks(m: torch.Tensor, rot, zoom, out_size: int) -> torch.Tensor:
    """B x C x 416 x 416 raw masks (-1 everywhere = absent) -> B x C x out x out: the image transform, threshold 0.5,
    absent channels stay -1 (the CROG object mask)."""
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
            if self.norm == "clip" and len(y) >= 5:  # CROG object mask: same transform, threshold 0.5, -1 = no mask
                y[4] = transform_masks(y[4], rot, zoom, self.out_size)
            yield normalise(x, self.norm), y, didx, rot, zoom, prompt, query
