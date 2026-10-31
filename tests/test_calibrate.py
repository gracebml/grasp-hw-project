import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("clip")


def load():
    p = Path(__file__).resolve().parents[1] / "experiments" / "calibrate.py"
    spec = importlib.util.spec_from_file_location("calibrate", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mean_pose_reads_grasp_pixels_only():
    pos = torch.zeros(1, 1, 2, 2)
    pos[0, 0, 0, 0] = 1.0
    cos, sin, width = torch.full((1, 1, 2, 2), 9.0), torch.full((1, 1, 2, 2), 9.0), torch.full((1, 1, 2, 2), 9.0)
    cos[0, 0, 0, 0], sin[0, 0, 0, 0], width[0, 0, 0, 0] = 0.5, -0.5, 0.25
    assert load().mean_pose([(None, (pos, cos, sin, width))]) == pytest.approx([0.5, -0.5, 0.25])
