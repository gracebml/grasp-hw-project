# Figures

The two figures of the paper, rebuilt from their data. Every script writes into `example/`; run them from the repository root.

| Figure | Script | Input |
|:---|:---|:---|
| Fig. 1, architecture | `run_example.py` → `make_tiles.py`, `make_ot_tile.py` → `grat_arch.tex` | `data/example_image.png`, `data/example_mask.png`, a checkpoint |
| Fig. 2, validation curves | `plot_val_curves.py` | `data/val_curves.json` |

## Figure 1: architecture

<img src="../assets/grat_arch.png" alt="GRAT architecture" width="640">

The tiles are real outputs of GG-CNN + GRAT (seed 123) on one Grasp-Anything validation image with the instruction
*"Lift spoon by its handle."*

```bash
# 1. maps of the network: relevance map r, grasp maps, decoded grasp (CPU is enough)
CKPT=checkpoints/grat/lggcnn_s123/epoch_11_iou_0.42 python visualization/run_example.py
# 2. image tiles, and the OT tile with the flows of the Sinkhorn plans
python visualization/make_tiles.py
python visualization/make_ot_tile.py
# 3. the figure (TikZ, needs the Inter font)
cd visualization && xelatex grat_arch.tex
```

`example/` already holds the outputs of steps 1 and 2, so step 3 works without a checkpoint.

The OT panel is an illustration. It shows the mass of the relevance map pooled to the 28 × 28 grid, the spoon's ground-truth
mask as the target, the cup as another object, and the flows that do the most transport work in the Sinkhorn plans computed by
`lgdbench.ot`. In training, the loss acts on the predicted grasp map instead.

## Figure 2: validation curves

```bash
python visualization/plot_val_curves.py               # GRAT with and without the OT loss
BASELINE=1 python visualization/plot_val_curves.py    # also the baselines
```

`data/val_curves.json` holds the validation success of each epoch, read from the training logs (seeds 123 and 456).
