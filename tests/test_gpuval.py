import numpy as np
import pytest

torch = pytest.importorskip("torch")
skf = pytest.importorskip("skimage.filters")
from skimage.feature import peak_local_max  # noqa: E402

from lgdbench import gpuval  # noqa: E402


def _maps(seed, n=24):
    # smooth maps with several bumps, like network outputs; some below the 0.2 threshold, one constant
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224]
    out = []
    for i in range(n):
        m = np.zeros((224, 224))
        for _ in range(rng.integers(1, 6)):
            cy, cx, s, a = rng.uniform(0, 224), rng.uniform(0, 224), rng.uniform(3, 25), rng.uniform(0.05, 1.0)
            m += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        m += 0.02 * rng.standard_normal((224, 224))
        out.append(m)
    out[-1] = np.full((224, 224), 0.5)
    return np.stack(out).astype(np.float32)


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_gaussian_matches_skimage(sigma):
    x = _maps(0)
    ref = np.stack([skf.gaussian(m, sigma, preserve_range=True) for m in x])
    ours = gpuval.gaussian(torch.from_numpy(x), sigma).numpy()
    assert ours.dtype == ref.dtype == np.float32
    assert np.abs(ours - ref).max() < 1e-6


def test_top_peak_matches_peak_local_max():
    x = _maps(1, 64)
    q = np.stack([skf.gaussian(m, 2.0, preserve_range=True) for m in x])
    found, rows, cols = gpuval.top_peak(torch.from_numpy(q))
    for i in range(len(q)):
        ref = peak_local_max(q[i], min_distance=20, threshold_abs=0.2, num_peaks=1)
        assert bool(found[i]) == (len(ref) > 0)
        if len(ref):
            assert (int(rows[i]), int(cols[i])) == tuple(ref[0])
