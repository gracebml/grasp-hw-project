"""OT tile of the architecture figure, drawn on the real image: the mass softmax(pooled r / tau) of the relevance map, the
target (the spoon's ground-truth mask), another object (the cup) and the flows doing the most transport work in the real
Sinkhorn plans of lgdbench.ot. An illustration: training applies the loss to the predicted grasp map.

    python visualization/make_ot_tile.py      ->  example/ot.png, example/ot_flows.tex
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from lgdbench import ot  # noqa: E402

OUT = HERE / "example"
Z, G = 448, ot.GRID
TAU, EPS, ITERS = 0.1, 0.05, 200
TARGET, OTHER, MASS = (40, 135, 80), (215, 95, 70), (16, 92, 104)
CROP = (60, 0, 448, 388)   # x0, y0, x1, y1 of the square kept around the spoon and the cup (pixels of the 448 tile)

d = np.load(OUT / "maps.npz")
rgb = Image.open(HERE / "data" / "example_image.png").convert("RGB").resize((Z, Z), Image.BICUBIC)
spoon = np.asarray(Image.open(HERE / "data" / "example_mask.png").convert("L").resize((Z, Z), Image.BILINEAR)) > 128
lum = np.asarray(rgb.convert("L")).astype(float)
yy, xx = np.mgrid[0:Z, 0:Z] * (224 / Z)
cup = (lum > 108) & (yy < 98) & (xx >= 75) & (xx < 190) & ~((yy >= 78) & (xx >= 140))   # table: luminance 62-96
cup = ndimage.binary_fill_holes(ndimage.binary_closing(cup, iterations=6))
cup = ndimage.binary_opening(cup, iterations=3)

# distributions on the 28 x 28 grid, as in training (lgdbench.ot)
rel = torch.from_numpy(d["rel"].astype(np.float32))[None, None]
log_p = ot.log_action_dist(rel, G, TAU)
pool = lambda m: torch.from_numpy(m.astype(np.float32))[None, None]  # noqa: E731
q_pos, _ = ot.target_dist(pool(spoon), G)
q_neg, _ = ot.target_dist(pool(cup), G)
cost = ot.grid_cost(G)


def plan(q):
    """Sinkhorn plan in the log domain of ot.sinkhorn_cost, returned as an N x N matrix."""
    log_b = torch.log(q.clamp_min(0)).masked_fill(q <= 0, float("-inf"))
    k = -cost / EPS
    f, g = torch.zeros_like(log_p), torch.zeros_like(log_p)
    for _ in range(ITERS):
        f = log_p - torch.logsumexp(k[None] + g[:, None, :], dim=2)
        g = log_b - torch.logsumexp(k[None] + f[:, :, None], dim=1)
        g = torch.where(torch.isfinite(g), g, torch.full_like(g, -1e9))
    return (k[None] + f[:, :, None] + g[:, None, :]).exp()[0]


def flows(P, k, min_cells, spread=4.0):
    """The k source cells that carry the most transport work (mass x distance to the mean destination of their mass), at
    least ``min_cells`` apart: (row, col) from, mean destination, share of the largest work."""
    P = P.numpy()
    rows, cols = np.divmod(np.arange(G * G), G)
    mass = P.sum(1)
    dest = (P @ np.stack([rows, cols], 1)) / np.maximum(mass, 1e-12)[:, None]
    dist = np.hypot(dest[:, 0] - rows, dest[:, 1] - cols)
    work = mass * dist
    out = []
    for i in np.argsort(work)[::-1]:
        if dist[i] < min_cells:
            continue
        if any(np.hypot(rows[i] - o[0] + .5, cols[i] - o[1] + .5) < spread for o in out):
            continue
        out.append((rows[i] + .5, cols[i] + .5, dest[i, 0] + .5, dest[i, 1] + .5, float(work[i])))
        if len(out) == k:
            break
    return out


pos, neg = plan(q_pos), plan(q_neg)
w_pos = float((pos * cost).sum()); w_neg = float((neg * cost).sum())
print("W+ %.3f  W- %.3f  (W+ < W- %s)" % (w_pos, w_neg, w_pos < w_neg))
f_pos, f_neg = flows(pos, 2, 3.0), flows(neg, 1, 3.0)

# tile: dimmed photo, light tints and thin outlines of the two objects, the mass as a glow; cropped around them
base = np.asarray(rgb).astype(float)
img = np.asarray(rgb.convert("L").convert("RGB")).astype(float) * .5 + 118


def soft(m, r):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))).astype(float)[..., None] / 255


def outline(img, m, col, alpha_fill):
    img = img * (1 - alpha_fill * soft(m, 2)) + np.array(col, float) * alpha_fill * soft(m, 2)
    edge = m ^ ndimage.binary_erosion(m, iterations=3)
    e = soft(edge, 1.2) * .95
    return img * (1 - e) + np.array(col, float) * e


img = outline(img, spoon, TARGET, .30)
img = outline(img, cup, OTHER, .28)
mass = log_p.exp()[0].reshape(G, G).numpy()
glow = np.asarray(Image.fromarray((mass / mass.max()).astype(np.float32)).resize((Z, Z), Image.BICUBIC)).clip(0, 1)
glow = soft(glow, 5)
al = np.clip(glow ** .6 * 1.25, 0, .9)
img = img * (1 - al) + np.array(MASS, float) * al
x0, y0, x1, y1 = CROP
Image.fromarray(img.clip(0, 255).astype(np.uint8)[y0:y1, x0:x1]).resize((Z, Z), Image.LANCZOS).save(OUT / "ot.png")


def to_tile(row, col):
    """Grid cell centre -> tile coordinates in [0, 1], x to the right and y downwards."""
    px, py = col * Z / G, row * Z / G
    return (px - x0) / (x1 - x0), (py - y0) / (y1 - y0)


with open(OUT / "ot_flows.tex", "w") as fh:
    fh.write("% largest-work flows of the Sinkhorn plans, tile coordinates x0/y0/x1/y1 in [0, 1] (y downwards), and work share\n")
    for name, fl in (("pos", f_pos), ("neg", f_neg)):
        top = max(x[4] for x in fl)
        items = ",".join("%.3f/%.3f/%.3f/%.3f/%.2f" % (*to_tile(r0, c0), *to_tile(r1, c1), w / top) for r0, c0, r1, c1, w in fl)
        fh.write("\\def\\flows%s{%s}\n" % (name, items))
for name, fl in (("pos", f_pos), ("neg", f_neg)):
    for f in fl:
        print(name, "from (%.1f, %.1f) to (%.1f, %.1f) mass %.3f" % f)
