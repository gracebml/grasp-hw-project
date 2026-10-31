"""Measure the two constants of GRAT before training: the pose prior and the OT weight.

    pose prior   mean ground-truth (cos 2theta, sin 2theta, width) on the grasp pixels of the training stream as the network
                 sees it, over N_PRIOR batches of 8 (1,600 images) -> grat.POSE_PRIOR
    lambda       0.5 |g_flow| / |g_OT| at initialisation, averaged over N_BATCHES -> grat.LAMBDA

Parameters (env): DATA_ROOT, SPLIT_LIST (default data_index/explore_train.txt), NETWORKS (default all three), N_PRIOR (200),
N_BATCHES (10), CLIP_RN50, OUT_DIR. Writes calibration.json.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lgdbench import data, gpuaug, grat, refs  # noqa: E402


def mean_pose(batches) -> list:
    """Mean cos, sin, width over the pixels with positive quality; ``batches`` yield the (x, y, ...) of GPUAugLoader."""
    import torch
    total, count = None, 0.0
    for b in batches:
        y_pos, y_cos, y_sin, y_width = b[1][:4]
        m = (y_pos > 0).double()
        s = (torch.cat([y_cos, y_sin, y_width], 1).double() * m).sum(dim=(0, 2, 3))
        total = s if total is None else total + s
        count += float(m.sum())
    return (total / count).tolist()


def lambda_rule(net, batches, cfg: grat.Config) -> float:
    """0.5 mean(|g_flow| / |g_OT|); same seed twice per batch, g_OT = g(lambda = 1) - g(lambda = 0)."""
    import numpy as np
    import torch
    params = [q for q in net.parameters() if q.requires_grad]
    ratios = []
    for j, b in enumerate(batches):
        grads = []
        for lam in (0.0, 1.0):
            net.cfg = dataclasses.replace(cfg, lam=lam)
            net.zero_grad()
            torch.manual_seed(1000 + j)
            grads.append(torch.autograd.grad(net.compute_loss(b[0], b[1], b[5], b[6])["loss"], params, allow_unused=True))
        g0, g1 = grads
        flow = torch.sqrt(sum((a ** 2).sum() for a in g0 if a is not None))
        transport = torch.sqrt(sum(((c - a) ** 2).sum() for a, c in zip(g0, g1) if a is not None and c is not None))
        if float(transport) > 0:
            ratios.append(float(flow / transport))
    return 0.5 * float(np.mean(ratios))


def main():
    import numpy as np
    import torch
    import torch.utils.data

    refs.add_lgd_to_path()
    root = Path(os.environ["DATA_ROOT"]).resolve()
    split = Path(os.environ.get("SPLIT_LIST", REPO / "data_index" / "explore_train.txt")).resolve()
    out = Path(os.environ.get("OUT_DIR", "outputs/local")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    clip_rn50 = os.path.expanduser(os.environ.get("CLIP_RN50", "~/.cache/clip/RN50.pt"))
    device = torch.device("cuda")

    def stream(mask_dir, norm, other_object):
        ds = data.make_gpuaug_dataset_class(split, mask_dir=mask_dir, other_object=other_object)(
            str(root / "grasp-anything"), output_size=224, ds_rotate=0.0, random_rotate=True, random_zoom=True,
            include_depth=0, include_rgb=1, seen=1, add_file_path=str(root / "grasp-anything++"))
        n = int(np.floor(0.9 * ds.length))
        ds.split_pools = (set(range(n)), set(range(n, ds.length)))
        idx = [i for i in range(n) if mask_dir is None or ds.has_object_mask(i)]
        torch.manual_seed(123)
        ld = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(idx))
        return gpuaug.GPUAugLoader(ld, device, norm)

    def take(it, k):
        return [b for b, _ in zip(it, range(k))]

    res = {"pose_prior": mean_pose(take(stream(None, "lgd", False), int(os.environ.get("N_PRIOR", "200"))))}
    print("pose prior", res["pose_prior"], flush=True)
    cfg = grat.Config(lam=1.0, pose_prior=tuple(res["pose_prior"]))
    nb = int(os.environ.get("N_BATCHES", "10"))
    for network in os.environ.get("NETWORKS", "lgrconvnet3,lggcnn,crog").split(","):
        mask_dir = str(root / "grasp-anything" / "mask") if network == "crog" else None
        batches = take(stream(mask_dir, "clip" if network == "crog" else "lgd", True), nb)
        torch.manual_seed(123)
        net = grat.build(network, cfg, clip_rn50)[0].to(device).train()
        res[f"lambda_{network}"] = lambda_rule(net, batches, cfg)
        print(network, res[f"lambda_{network}"], flush=True)
        del net
    (out / "calibration.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
