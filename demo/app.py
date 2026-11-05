"""Streamlit demo of GRAT. Run from this folder: ``cd demo && streamlit run app.py``."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

EXAMPLE_IMAGE = P.REPO / "visualization" / "data" / "example_image.png"
EXAMPLE_TEXT = "Lift spoon by its handle."
TILE = 300
NAVY, TEAL, AMBER = "#1a3a4a", "#2d8b8b", "#e3a21a"
CYAN, MAGENTA = "#12e2ea", "#ec1fe4"     # grasp rectangle, as in the LGD paper
SEQ = LinearSegmentedColormap.from_list("grat", ["#0f2530", NAVY, TEAL, "#8fd0c4", "#eef8f5"])
CYC = colormaps["twilight_shifted"]   # light at angle 0
RANGES = {"q": (0.0, 1.0, SEQ), "angle": (-np.pi / 2, np.pi / 2, CYC), "width": (0.0, 150.0, SEQ)}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }
.block-container { max-width: 1240px; padding-top: 2.2rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 2.1rem; margin-bottom: 0.1rem; }
.lede { font-size: 1.05rem; line-height: 1.55; max-width: 68ch; color: #3d5a66; margin-bottom: 1.4rem; }
.step { display: flex; align-items: baseline; gap: 0.7rem; margin: 2.2rem 0 0.3rem; }
.step .n { flex: none; width: 1.9rem; height: 1.9rem; border-radius: 50%; background: #2d8b8b; color: #fff;
           display: inline-flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.95rem;
           transform: translateY(-0.15rem); }
.step h3 { margin: 0; padding: 0; font-weight: 600; font-size: 1.3rem; }
.why { color: #3d5a66; max-width: 72ch; line-height: 1.55; margin: 0 0 0.9rem 2.6rem; }
.nums { font-variant-numeric: tabular-nums; color: #3d5a66; font-size: 0.9rem; }
[data-testid="stImageCaption"], .stCaption { font-variant-numeric: tabular-nums; }
@media (max-width: 640px) { .why { margin-left: 0; } }
</style>
"""


# ---------- drawing ----------
def colorize(a: np.ndarray, key: str, size: int = TILE) -> Image.Image:
    lo, hi, cmap = RANGES[key]
    rgb = (cmap(np.clip((a - lo) / (hi - lo), 0, 1))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.BILINEAR)


def angle_tile(ang: np.ndarray, q: np.ndarray, size: int = TILE) -> Image.Image:
    """Hue = grasp angle, brightness = quality: the angle only matters where a grasp is likely."""
    lo, hi, cmap = RANGES["angle"]
    hue = cmap(np.clip((ang - lo) / (hi - lo), 0, 1))[..., :3]
    m = np.clip(q, 0, 1)[..., None]
    bg = np.array([15, 37, 48]) / 255.0
    rgb = ((hue * m + bg * (1 - m)) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.BILINEAR)


def overlay(img: np.ndarray, r: np.ndarray, size: int = TILE) -> Image.Image:
    base = Image.fromarray(img).resize((size, size), Image.BICUBIC)
    return Image.blend(base, colorize(r, "q", size), 0.6)


def _dashed(d: ImageDraw.ImageDraw, a, b, fill, width: int, dash: float = 7.0) -> None:
    (x0, y0), (x1, y1) = a, b
    n = max(1, int(np.hypot(x1 - x0, y1 - y0) / dash))
    for k in range(0, n, 2):
        u, v = k / n, min(k + 1, n) / n
        d.line([(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u), (x0 + (x1 - x0) * v, y0 + (y1 - y0) * v)], fill=fill, width=width)


def draw_grasps(img: np.ndarray, grasps, size: int = TILE, candidate=None, ss: int = 3) -> Image.Image:
    """Grasp rectangles in the style of the LGD paper: thin cyan sides along the gripper opening, magenta jaws.
    Corners come from LGD's ``Grasp.as_gr`` (row, col): p0-p1 and p2-p3 span the opening, p1-p2 and p3-p0 are the
    jaws; an amber dot marks the centre. ``candidate``: the below-threshold position of highest quality, dashed. Drawn at ``ss`` x size, then
    downsampled, for smooth edges."""
    big = size * ss
    im = Image.fromarray(img).resize((big, big), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    s = big / P.SIZE
    items = [(g, 3 if i == 0 else 2, d.line) for i, g in enumerate(grasps)]
    if candidate is not None:
        items.append((candidate, 3, lambda xy, fill, width: _dashed(d, *xy, fill=fill, width=width, dash=6.0 * ss)))
    for g, px, line in reversed(items):        # the top grasp drawn last, on top
        p = [(c * s, r * s) for r, c in g[6]]
        w = px * ss
        line([p[0], p[1]], fill=CYAN, width=w)
        line([p[2], p[3]], fill=CYAN, width=w)
        line([p[1], p[2]], fill=MAGENTA, width=w)
        line([p[3], p[0]], fill=MAGENTA, width=w)
        cx, cy, r = g[1] * s, g[0] * s, (px + 2) * ss       # centre dot
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=AMBER, outline=NAVY, width=ss)
    return im.resize((size, size), Image.LANCZOS)


def no_grasp_caption(r: P.Result) -> str:
    return f"Best candidate: quality {r.best[5]:.2f}, below {P.THRESHOLD}, so no grasp"


def framed(im: Image.Image, on: bool) -> Image.Image:
    return ImageOps.expand(im, border=4, fill=TEAL if on else "#f4f7f7")


def step(n: int | None, title: str, why: str) -> None:
    num = f'<span class="n">{n}</span>' if n is not None else ""
    st.markdown(f'<div class="step">{num}<h3>{title}</h3></div><p class="why">{why}</p>', unsafe_allow_html=True)


def map_row(maps: dict, img: np.ndarray, at: str = "") -> None:
    """Quality over the image, angle coloured by quality, opening width."""
    q, ang, w = maps["q"], maps["angle"], maps["width"]
    c1, c2, c3 = st.columns(3)
    c1.image(overlay(img, q), caption=f"Quality{at} over the image, range {q.min():.2f} to {q.max():.2f}")
    c2.image(angle_tile(ang, q), caption=f"Angle{at}, shown where quality is high")
    c3.image(colorize(w, "width"), caption=f"Opening width{at}, {w.min():.0f} to {w.max():.0f} px")


# ---------- model ----------
@st.cache_resource(max_entries=2, show_spinner=False)
def model(path: str):
    return P.load_model(path)


@st.cache_data(max_entries=16, show_spinner=False)
def infer(path: str, img_bytes: bytes, text: str, n_grasps: int) -> P.Result:
    rgb = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return P.run(model(path), rgb, text, n_grasps, ckpt=path)


def grasp_table(res: P.Result) -> None:
    rows = [{"Grasp": i + 1, "Centre (x, y)": f"({g[1]}, {g[0]})", "Angle (deg)": round(np.degrees(g[2]), 1),
             "Opening (px)": round(g[3], 1), "Jaw size (px)": round(g[4], 1), "Quality": round(g[5], 3)}
            for i, g in enumerate(res.grasps)]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def timings(res: P.Result) -> str:
    return ", ".join(f"{k} {v:.2f} s" for k, v in res.timings.items())


# ---------- page ----------
st.set_page_config(page_title="GRAT demo", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

ckpts = P.list_checkpoints()
if not ckpts:
    st.error(f"No checkpoint found under {P.CKPT_ROOT}. Download the released checkpoints (link in the main README) into "
             "checkpoints/baseline and checkpoints/grat, or set GRAT_CKPT_ROOT to their folder.")
    st.stop()

with st.sidebar:
    st.header("Model")
    nets = [n for n in P.NETWORKS if any(c.network == n for c in ckpts)]
    network = st.radio("Pipeline", nets, format_func=P.NETWORKS.get)
    groups = [g for g in P.GROUPS if any(c.network == network and c.group == g for c in ckpts)]
    group = st.radio("Method", groups, index=groups.index("grat") if "grat" in groups else 0, format_func=P.GROUPS.get,
                     horizontal=True)
    seeds = sorted(c.seed for c in ckpts if c.network == network and c.group == group)
    seed = st.selectbox("Seed", seeds)
    ck = next(c for c in ckpts if (c.network, c.group, c.seed) == (network, group, seed))
    epoch, iou = P.CKPT_NAME.search(ck.path.name).groups()
    st.caption(f"Checkpoint of epoch {int(epoch)}, with validation success {float(iou):.2f} in its file name.")
    other_group = "baseline" if group == "grat" else "grat"
    other = next((c for c in ckpts if (c.network, c.group, c.seed) == (network, other_group, seed)), None)
    compare = st.toggle(f"Compare with {'the baseline' if other_group == 'baseline' else 'GRAT'}", value=False,
                        disabled=other is None)
    n_grasps = st.slider("Grasps to draw", 1, 5, 1, help="The success metric scores the first grasp only.")

st.title("GRAT: Grounding as Transport")
st.markdown('<p class="lede">Language-driven grasp detection. Give an image and an instruction naming an object part. '
            "CLIP first guesses where the instruction points; GRAT then carries that guess, together with the grasp "
            "pose, to the final grasp in eight flow steps.</p>", unsafe_allow_html=True)

with st.form("inputs", border=False):
    left, right = st.columns([1, 1.3], gap="large")
    upload = left.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"],
                                help="Non-square images are cropped to their centre square, then resized to 224 x 224.")
    text = right.text_area("Instruction", EXAMPLE_TEXT, height=100,
                           help="Name the object and the part to hold, as in Grasp-Anything++.")
    submitted = right.form_submit_button("Detect grasp", type="primary")

img_bytes = upload.getvalue() if upload is not None else EXAMPLE_IMAGE.read_bytes()
if upload is None:
    left.caption("No image uploaded: using the example image of the paper.")

if submitted:
    if not text.strip():
        st.error(f"Write an instruction first, for example “{EXAMPLE_TEXT}”")
        st.stop()
    st.session_state.job = (img_bytes, text.strip())
if "job" not in st.session_state:          # first visit: show the paper's example
    st.session_state.job = (EXAMPLE_IMAGE.read_bytes(), EXAMPLE_TEXT)

job_bytes, job_text = st.session_state.job
with st.spinner(f"Running {ck.label}"):
    res = infer(str(ck.path), job_bytes, job_text, n_grasps)
    res_other = infer(str(other.path), job_bytes, job_text, n_grasps) if compare and other else None

st.markdown(f'<p class="nums">{ck.label}. Instruction: “{job_text}”<br>Time on CPU: {timings(res)}</p>',
            unsafe_allow_html=True)

if res.is_grat:
    step(1, "Relevance map r",
         "A frozen CLIP ViT-B/16 compares the instruction with every image patch. Bright areas are where CLIP thinks the "
         "instruction points. This coarse guess is the starting point of the flow, in place of random noise.")
    c1, c2, _ = st.columns(3)
    c1.image(Image.fromarray(res.image).resize((TILE, TILE), Image.BICUBIC), caption="Input, 224 x 224")
    c2.image(overlay(res.image, res.relevance), caption="r over the image")

    step(2, "Source x₀",
         "r becomes the starting quality map. The pose channels start from a measured pose prior where r is high and "
         "from the no-grasp background where r is low.")
    map_row(res.states[0], res.image, " at t = 0")

    step(3, "Flow from the guess to the grasp",
         "A rectified flow moves all four maps at once with eight Euler steps, x ← x + v(x, t) / 8. "
         "Drag the slider to watch the quality mass leave the wrong regions and settle on the named part.")
    k = st.slider("Step", 0, len(res.states) - 1, len(res.states) - 1, format="%d of 8")
    map_row(res.states[k], res.image, f" at t = {k}/8")
    strip = st.columns(len(res.states))
    for i, (col, s) in enumerate(zip(strip, res.states)):
        col.image(framed(colorize(s["q"], "q", 120), i == k), caption=f"t = {i}/8", use_column_width=True)
    out_n = 4
else:
    step(None, "One forward pass",
         "The baseline regresses the four grasp maps directly from the image and the instruction. It has no relevance "
         "map, source or flow; switch the method to + GRAT to see them.")
    out_n = None

step(out_n, "Grasp",
     "LGD's post-processing smooths the maps, then its peak detector picks the grasp centres with quality above 0.2. "
     "In each rectangle, the cyan sides span the gripper opening and the magenta sides are the jaws.")
g1, g2, g3, g4 = st.columns(4)
if res.grasps:
    g1.image(draw_grasps(res.image, res.grasps), caption="Detected grasp")
else:
    g1.image(draw_grasps(res.image, [], candidate=res.best), caption=no_grasp_caption(res))
g2.image(colorize(res.q, "q"), caption=f"Quality, range {res.q.min():.2f} to {res.q.max():.2f}")
g3.image(angle_tile(res.angle, res.q), caption="Angle, shown where quality is high")
g4.image(colorize(res.width, "width"), caption=f"Opening width, {res.width.min():.0f} to {res.width.max():.0f} px")
if res.grasps:
    grasp_table(res)
else:
    st.warning(f"No grasp. The highest quality is {res.best[5]:.2f}, below the {P.THRESHOLD} threshold of LGD's detector, "
               "so the official metric counts this sample as a failure. The dashed rectangle only shows where that "
               "highest quality lies. Try an instruction written like those of Grasp-Anything++, such as “Grasp the cup "
               "by its handle.”, or another seed or pipeline.")

if res_other is not None:
    st.subheader("Baseline and GRAT on the same input")
    pair = sorted([(ck, res), (other, res_other)], key=lambda p: p[0].group)  # baseline first
    for col, (c, r) in zip(st.columns(2, gap="large"), pair):
        a, b = col.columns(2)
        a.image(draw_grasps(r.image, r.grasps, candidate=r.best),
                caption=c.label if r.grasps else f"{c.label}. {no_grasp_caption(r)}")
        b.image(colorize(r.q, "q"), caption="Quality after post-processing")

st.divider()
st.caption("This page illustrates the method on single images. Images outside Grasp-Anything are out of distribution, "
           "and nothing here is an evaluation; the paper's numbers come from experiments/evaluate.py.")
