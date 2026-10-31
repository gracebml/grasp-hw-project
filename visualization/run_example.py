import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from lgdbench import grat, refs, relevance  # noqa: E402

refs.add_lgd_to_path()
from inference.post_process import post_process_output  # noqa: E402
from utils.dataset_processing.grasp import detect_grasps  # noqa: E402

QUERY = "Lift spoon by its handle."


def main():
    grat.accept_released_checkpoints()
    net = torch.load(os.environ["CKPT"], map_location="cpu", weights_only=False).float().eval()
    rgb = Image.open(HERE / "data" / "example_image.png").convert("RGB").resize((224, 224), Image.BICUBIC)
    im = np.asarray(rgb).astype(np.float32) / 255
    im -= im.mean()                                   # LGD normalisation: /255 minus the image mean
    xc = torch.from_numpy(im.transpose(2, 0, 1))[None]
    with torch.no_grad():
        rel = net.relevance(relevance.rgb_from_lgd(xc), [QUERY])
        x = grat.source_maps(rel, net.cfg, sample=False)
        for k in range(net.cfg.steps):
            t = grat.time_grid(k, net.cfg.steps, xc)
            x = x + torch.cat(net(net.inputs(xc, x, t), [QUERY], [QUERY]), 1) / net.cfg.steps
    q, ang, w = post_process_output(*x.split(1, 1))
    g = detect_grasps(q, ang, width_img=w, no_grasps=1)[0]
    (HERE / "example").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "example" / "maps.npz", rel=rel.numpy()[0, 0], q=q, ang=ang, w=w,
             grasp=np.array([*g.center, g.angle, g.length, g.width]))
    print("grasp centre", g.center, "angle %.3f" % g.angle)


if __name__ == "__main__":
    main()
