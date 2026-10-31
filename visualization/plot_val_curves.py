import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = json.loads((HERE / "data" / "val_curves.json").read_text())
OUT = HERE / "example"
STYLE = {  # palette validated with the dataviz validator (CVD dE 24.7, contrast >= 3:1)
    "GRAT": dict(color="#2a78d6", label="GRAT"),
    "noOT": dict(color="#eb6834", label="GRAT w/o OT"),
    "base": dict(color="#8a8a8a", label="baseline", ls=(0, (3, 2))),
}
KINDS = ("base", "noOT", "GRAT") if os.environ.get("BASELINE", "0") == "1" else ("noOT", "GRAT")
NAME = "val_curves_baseline" if os.environ.get("BASELINE", "0") == "1" else "val_curves"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})


def curves(pipe, kind):
    runs = [DATA[f"{pipe}/{kind}/s{s}"] for s in (123, 456)]
    n = max(len(r["val"]) for r in runs)
    m = np.full((len(runs), n), np.nan)
    for i, r in enumerate(runs):
        m[i, :len(r["val"])] = r["val"]
    return np.arange(n), np.nanmean(m, 0), np.nanmin(m, 0), np.nanmax(m, 0)


fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.12), sharey=True)
for ax, (pipe, title) in zip(axes, [("GR", "GR-ConvNet"), ("GG", "GG-CNN")]):
    for kind in KINDS:
        st = STYLE[kind]
        x, mean, lo, hi = curves(pipe, kind)
        if kind != "base":
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, lw=0)
        ax.plot(x, mean, color=st["color"], lw=1.0, ls=st.get("ls", "-"), label=st["label"])
        if pipe == "GR" and kind == "noOT":  # collapsed runs stopped early
            ax.plot(x[-1], mean[-1], marker="x", ms=3.5, mew=0.9, color=st["color"])
    ax.set_title(title, pad=2)
    ax.set_xlim(0, 49)
    ax.set_ylim(0, 0.47)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Epoch", labelpad=1)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Val. success", labelpad=1)
h, l = axes[0].get_legend_handles_labels()
order = list(range(len(h)))[::-1]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.55, 1.03), handlelength=1.6, columnspacing=1.2)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.29, top=0.72, wspace=0.08)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / f"{NAME}.pdf")
fig.savefig(OUT / f"{NAME}.png", dpi=300)
print("saved", OUT / f"{NAME}.pdf")
