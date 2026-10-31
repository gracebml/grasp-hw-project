from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CLIP_MEAN, CLIP_STD


def rgb_from_lgd(xc: torch.Tensor) -> torch.Tensor:
    """Undo the LGD normalisation (``Image.normalise``: /255 minus ONE scalar mean per image) up to that scalar,
    assuming the darkest value of the image is 0."""
    return (xc - xc.amin(dim=(1, 2, 3), keepdim=True)).clamp(0, 1)


def minmax(s: torch.Tensor) -> torch.Tensor:
    """Per-image min-max to [0, 1] of a B x 1 x H x W map."""
    lo = s.amin(dim=(1, 2, 3), keepdim=True)
    hi = s.amax(dim=(1, 2, 3), keepdim=True)
    return (s - lo) / (hi - lo).clamp_min(1e-6)


class ClipRelevance(nn.Module):
    """Min-max normalised cosine between the CLIP text embedding and the dense patch features: value path of the last
    attention block, no query-key attention, no residual or MLP. Output B x 1 x H x W in [0, 1]."""

    def __init__(self, name: str = "ViT-B/16"):
        super().__init__()
        import clip
        self.model, _ = clip.load(name, device="cpu", jit=False)  # fp32
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def dense(self, img: torch.Tensor) -> torch.Tensor:
        vis = self.model.visual
        x = vis.conv1(img)
        b, c, g, _ = x.shape
        x = x.reshape(b, c, g * g).permute(0, 2, 1)
        cls = vis.class_embedding + torch.zeros(b, 1, c, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], 1) + vis.positional_embedding
        x = vis.ln_pre(x).permute(1, 0, 2)
        for blk in vis.transformer.resblocks[:-1]:
            x = blk(x)
        last = vis.transformer.resblocks[-1]
        w, bias = last.attn.in_proj_weight, last.attn.in_proj_bias
        v = F.linear(last.ln_1(x), w[2 * c:], bias[2 * c:])
        z = vis.ln_post(last.attn.out_proj(v).permute(1, 0, 2))[:, 1:] @ vis.proj
        return F.normalize(z, dim=-1).reshape(b, g, g, -1)

    @torch.no_grad()
    def forward(self, rgb01: torch.Tensor, texts) -> torch.Tensor:
        import clip
        h, w = rgb01.shape[-2:]
        img = F.interpolate(rgb01, size=(224, 224), mode="bilinear", align_corners=False) if (h, w) != (224, 224) \
            else rgb01
        feat = self.dense((img - self.mean) / self.std)
        tok = clip.tokenize(list(texts), truncate=True).to(rgb01.device)
        t = F.normalize(self.model.encode_text(tok), dim=-1)
        s = torch.einsum("bghd,bd->bgh", feat, t)[:, None]
        return minmax(F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False))
