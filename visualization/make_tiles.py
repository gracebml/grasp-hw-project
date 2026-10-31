"""Image tiles of the architecture figure from the maps of run_example.py: input, relevance map r, grasp maps, grasp.

    python visualization/make_tiles.py      ->  example/{image,r,q,grasp}.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from lgdbench import refs  # noqa: E402

refs.add_lgd_to_path()
np.float = float  # the official grasp.py still uses this alias, removed in NumPy >= 1.24
from utils.dataset_processing.grasp import Grasp  # noqa: E402

Z = 448                                              # tile size in pixels
RELEVANCE = ((250, 185, 70), (232, 72, 30))          # colour ramp of r
QUALITY = ((120, 205, 200), (16, 92, 104))           # colour ramp of the grasp maps
GRASP = (20, 110, 120)


def main():
    out = HERE / "example"
    d = np.load(out / "maps.npz")
    rgb = Image.open(HERE / "data" / "example_image.png").convert("RGB").resize((Z, Z), Image.BICUBIC)
    rgb.save(out / "image.png")
    gray = np.asarray(rgb.convert("L").convert("RGB")).astype(float) * .55 + 90

    def norm(m, p=60):
        lo = np.percentile(m, p)
        return np.clip((m - lo) / (m.max() - lo + 1e-6), 0, 1)

    def over(m, ramp):
        """The map as a two-colour ramp over the dimmed image; opacity grows with the map."""
        m = np.clip(np.asarray(Image.fromarray(m.astype(np.float32)).resize((Z, Z), Image.BICUBIC)), 0, 1)[..., None]
        col = np.array(ramp[0], float) * (1 - m) + np.array(ramp[1], float) * m
        al = np.clip(m ** .8 * 1.25, 0, .92)
        return Image.fromarray((gray * (1 - al) + col * al).astype(np.uint8))

    over(norm(d["rel"]), RELEVANCE).save(out / "r.png")
    over(norm(d["q"]), QUALITY).save(out / "q.png")

    # the decoded grasp: the official rectangle, drawn at 4x then downsampled
    y, x, ang, length, width = d["grasp"]
    xy = [(float(c) * Z / 224, float(r) * Z / 224) for r, c in Grasp((y, x), ang, length, width).as_gr.points]
    s = 4
    g = rgb.resize((Z * s, Z * s), Image.BICUBIC)
    dr = ImageDraw.Draw(g)

    def seg(u, v, halo, core):
        pts = [(xy[u][0] * s, xy[u][1] * s), (xy[v][0] * s, xy[v][1] * s)]
        dr.line(pts, fill=(255, 255, 255), width=halo)
        dr.line(pts, fill=GRASP, width=core)

    for u, v in ((0, 1), (3, 2)):
        seg(u, v, 44, 24)
    for u, v in ((0, 3), (1, 2)):
        seg(u, v, 84, 60)
    g.resize((Z, Z), Image.LANCZOS).save(out / "grasp.png")


if __name__ == "__main__":
    main()
