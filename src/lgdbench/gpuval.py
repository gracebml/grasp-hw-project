"""Validation scoring on the GPU, used only to pick checkpoints during training.

The official ``train_network.validate()`` scores one sample at a time on the CPU: ``post_process_output`` (skimage
gaussian), ``detect_grasps`` (``peak_local_max``) and ``calculate_iou_match``. Here the first two run batched on the GPU,
reproducing skimage 0.24 / scipy 1.10 step by step; the IoU test stays the official ``Grasp.max_iou`` on the CPU for the
single top-1 grasp of each sample. Test evaluation keeps the official ``evaluate.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

WIDTH_SCALE = 150.0  # inference/post_process.py:16 (decode as the official code)


def gaussian(x: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """``skimage.filters.gaussian(x, sigma, preserve_range=True)`` for a B x H x W float32 batch.

    skimage keeps float32 and calls ``ndi.gaussian_filter`` with mode 'nearest': a separable correlation along axis 0
    and then axis 1, each pass computed in double and stored as float32."""
    radius = int(truncate * float(sigma) + 0.5)
    t = torch.arange(-radius, radius + 1, dtype=torch.float64, device=x.device)
    w = torch.exp(-0.5 * (t / sigma) ** 2)
    w = w / w.sum()
    y = F.pad(x.double()[:, None], (0, 0, radius, radius), mode="replicate")
    y = F.conv2d(y, w.view(1, 1, -1, 1))[:, 0].float()
    y = F.pad(y.double()[:, None], (radius, radius, 0, 0), mode="replicate")
    return F.conv2d(y, w.view(1, 1, 1, -1))[:, 0].float()


def post_process(pos, cos, sin, width):
    """``inference.post_process.post_process_output`` for B x 1 x H x W maps -> three B x H x W float32 maps."""
    q = gaussian(pos[:, 0].float(), 2.0)
    ang = gaussian((torch.atan2(sin, cos) / 2.0)[:, 0].float(), 2.0)
    w = gaussian(width[:, 0].float() * WIDTH_SCALE, 1.0)
    return q, ang, w


def top_peak(q: torch.Tensor, min_distance: int = 20, threshold_abs: float = 0.2):
    """``peak_local_max(q, min_distance, threshold_abs, num_peaks=1)`` per image (detect_grasps, grasp.py).

    Returns ``found`` (B,) and ``rows``, ``cols`` (B,) of the highest peak."""
    k = 2 * min_distance + 1
    qmax = F.max_pool2d(q[:, None], k, stride=1, padding=min_distance)[:, 0]  # = maximum_filter, mode 'nearest'
    cand = q == qmax
    trivial = cand.flatten(1).all(dim=1)  # skimage: no peak in a constant image
    cand &= q > threshold_abs
    b = min_distance
    cand[:, :b] = False
    cand[:, -b:] = False
    cand[:, :, :b] = False
    cand[:, :, -b:] = False
    cand &= ~trivial[:, None, None]
    score = torch.where(cand, q, torch.full_like(q, -math.inf))
    flat = score.flatten(1).argmax(dim=1)  # first maximum in row-major order = stable descending sort
    found = cand.flatten(1).any(dim=1)
    return found, flat // q.shape[2], flat % q.shape[2]


def validate(net, device, val_data, iou_threshold):
    """Same contract as the official ``train_network.validate`` (correct / failed / loss / losses)."""
    from utils.dataset_processing.grasp import Grasp

    net.eval()
    results = {"correct": 0, "failed": 0, "loss": 0, "losses": {}}
    n_total = len(val_data.loader.sampler)  # number of validation samples
    with torch.no_grad():
        for x, y, didx, rot, zoom, prompt, query in val_data:
            xc = x.to(device)
            yc = [t.to(device) for t in y]
            lossd = net.compute_loss(xc, yc, prompt, query)
            bs = xc.shape[0]
            results["loss"] += lossd["loss"].item() * bs / n_total
            for ln, l in lossd["losses"].items():
                results["losses"][ln] = results["losses"].get(ln, 0) + float(l) * bs / n_total
            p = lossd["pred"]
            q, ang, w = post_process(p["pos"], p["cos"], p["sin"], p["width"])
            found, rows, cols = top_peak(q)
            found, rows, cols = found.cpu().numpy(), rows.cpu().numpy(), cols.cpu().numpy()
            a = ang[torch.arange(bs), torch.as_tensor(rows), torch.as_tensor(cols)].cpu().numpy()
            wd = w[torch.arange(bs), torch.as_tensor(rows), torch.as_tensor(cols)].cpu().numpy()
            for i in range(bs):
                ok = False
                if found[i]:
                    g = Grasp((rows[i], cols[i]), a[i])  # detect_grasps: centre, angle, length = width, width = length / 2
                    g.length = wd[i]
                    g.width = g.length / 2
                    gt = val_data.dataset.get_gtbb(int(didx[i]), float(rot[i]), float(zoom[i]))
                    ok = g.max_iou(gt) > iou_threshold  # official IoU (calculate_iou_match, evaluation.py:76)
                results["correct" if ok else "failed"] += 1
    return results
