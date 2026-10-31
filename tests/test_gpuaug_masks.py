import math

import pytest

torch = pytest.importorskip("torch")
from lgdbench import gpuaug  # noqa: E402


def _old_mask_path(m, rot, zoom, out):
    # the single-channel object mask path, kept here as the reference
    missing = (m < 0).flatten(1).all(dim=1)
    m = torch.stack([gpuaug.transform(m[i], rot[i], zoom[i], out) for i in range(m.shape[0])])
    m = (m > 0.5).float()
    m[missing] = -1.0
    return m


def test_transform_masks_equals_old_single_channel_path():
    torch.manual_seed(0)
    m = (torch.rand(3, 1, 416, 416) > 0.7).float()
    m[1] = -1.0
    rot = [0.0, math.pi / 2, math.pi]
    zoom = [1.0, 0.8, 0.6]
    assert torch.equal(gpuaug.transform_masks(m, rot, zoom, 224), _old_mask_path(m, rot, zoom, 224))


def test_transform_masks_multi_channel_absent_channel():
    m = torch.zeros(2, 2, 416, 416)
    m[:, 0, :100, :100] = 1.0
    m[0, 1] = -1.0          # absent in sample 0 only
    out = gpuaug.transform_masks(m, [0.0, 0.0], [1.0, 1.0], 224)
    assert out.shape == (2, 2, 224, 224)
    assert (out[0, 1] == -1).all() and (out[1, 1] == 0).all()
    assert torch.equal(out[:, 0], _old_mask_path(m[:, :1], [0.0, 0.0], [1.0, 1.0], 224)[:, 0])
