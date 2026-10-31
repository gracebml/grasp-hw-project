import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")
pytest.importorskip("cv2")  # the official image.py imports it
from lgdbench import gpuaug, refs  # noqa: E402


def _scene():
    # smooth, asymmetric 416x416x3 uint8 image: a wrong rotation direction or crop would not match
    yy, xx = np.mgrid[0:416, 0:416] / 415.0
    blob = np.exp(-((xx - 0.7) ** 2 + (yy - 0.25) ** 2) / 0.01)
    img = np.stack([200 * xx + 40 * blob, 150 * yy ** 2 + 60 * blob, 100 * blob + 50 * xx * yy], -1)
    return img.clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("zoom", [1.0, 0.75, 0.52])
def test_transform_matches_official_get_rgb_ops(k, zoom):
    refs.add_lgd_to_path()
    from utils.dataset_processing import image

    rot = k * math.pi / 2
    raw = _scene()
    ref = image.Image(raw.copy())  # the official order in grasp_anywhere_data.get_rgb
    ref.rotate(rot)
    ref.zoom(zoom)
    ref.resize((224, 224))
    ref = ref.img.astype(np.float32) / 255.0

    x = torch.from_numpy(raw).permute(2, 0, 1).float() / 255.0
    ours = gpuaug.transform(x, rot, zoom).permute(1, 2, 0).numpy()

    assert ours.shape == ref.shape
    assert np.abs(ours - ref).mean() < 2 / 255  # interpolation / uint8 rounding only
    wrong = gpuaug.transform(x, rot + math.pi / 2, zoom).permute(1, 2, 0).numpy()
    assert np.abs(wrong - ref).mean() > 10 * np.abs(ours - ref).mean()


def test_fifth_target_is_transformed_as_a_mask_only_on_the_crog_branch():
    import torch
    from lgdbench import gpuaug

    x = torch.zeros(1, 8, 8, 3, dtype=torch.uint8)
    extra = torch.zeros(1, 1, 8, 8)
    extra[0, 0, 0, :] = 0.7                                   # an extra target: already in the output frame
    batch = [(x, [torch.zeros(1, 1, 4, 4)] * 4 + [extra], [0], torch.tensor([1.5707963]), torch.tensor([1.0]), ["p"], ["q"])]
    out = next(iter(gpuaug.GPUAugLoader(batch, torch.device("cpu"), "lgd", out_size=8)))
    assert torch.equal(out[1][4], extra)                      # untouched on the GR-ConvNet branch
    crog = next(iter(gpuaug.GPUAugLoader(batch, torch.device("cpu"), "clip", out_size=8)))
    assert not torch.equal(crog[1][4], extra)                 # CROG: rotated and thresholded as before
