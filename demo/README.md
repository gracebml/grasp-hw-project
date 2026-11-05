# GRAT demo

A Streamlit page that runs a released checkpoint on one image and one instruction, and shows each stage of the inference:

1. the CLIP relevance map r,
2. the source x₀ built from r and the pose prior,
3. the eight Euler steps of the flow (slider and filmstrip),
4. the grasp maps after LGD's post-processing and the detected grasp rectangle.

Baselines run in one forward pass, so they show stage 4 only. A toggle puts the baseline and GRAT of the same pipeline and
seed side by side.

The demo lives on its own: it imports `lgdbench` and the official LGD / CROG code read-only, and changes nothing outside
this folder.

## Run

```bash
conda create -n grat-demo python=3.10 -y && conda activate grat-demo
pip install "setuptools<70" wheel
pip install --no-build-isolation -r demo/requirements.txt    # CPU only; CLIP's setup.py needs pkg_resources
cd demo && streamlit run app.py                               # http://localhost:8501
```

The checkpoints are read from `checkpoints/{baseline,grat}/<network>_s<seed>/` (see the main README), or from
`GRAT_CKPT_ROOT`. Everything runs on CPU: loading a checkpoint takes about 1 s, inference 0.1 s (baseline) to 2.5 s (CROG + GRAT).

## Files

| File | Content |
|:---|:---|
| `pipeline.py` | checkpoint list, preprocessing, inference with every Euler state kept; no Streamlit |
| `app.py` | the page |
| `tests/test_pipeline.py` | preprocessing, checkpoint list; with a checkpoint: the maps of `visualization/example/maps.npz` are reproduced, and the unrolled steps end at GRAT's own `compute_loss` |
| `.streamlit/config.toml` | theme; the server listens on localhost only |
| `space/` | Dockerfile, start-up script and card of the Hugging Face Space |

```bash
python -m pytest demo/tests
```

## Deploy (Hugging Face Space)

`space/` holds a Docker Space: the image clones this repository at a pinned commit with its submodules and installs
`requirements.txt`; at start-up, `entrypoint.sh` downloads the checkpoints from a Hugging Face model repository
(`CKPT_REPO`, laid out as `baseline/` and `grat/` above) and serves the page on port 7860. The deployed Space offers the
seed-123 checkpoints.

## Notes

- Preprocessing follows the official loader: resize to 224 × 224 with LGD's `Image.resize`, then LGD's normalisation (CLIP's
  for CROG). Non-square uploads are first cropped to their centre square; this step is the demo's own.
- Images outside Grasp-Anything are out of distribution. The page illustrates the method; it is not an evaluation.
- On file systems without inotify (e.g. NTFS over FUSE), Streamlit does not see edits to `app.py`: restart the server.
