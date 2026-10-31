"""Runtime fixes for defects of the official code.

No file of the reference code is edited: the official functions are wrapped before the official script imports them.
"""
from __future__ import annotations

from . import refs


def no_detach_baselines() -> None:
    """Route ``get_network('lgrconvnet3' | 'lggcnn')`` to the fixed subclasses; every other name is unchanged."""
    refs.add_lgd_to_path()
    import inference.models as official

    from . import fixed_models

    original = getattr(official.get_network, "__wrapped__", official.get_network)
    fixed = {"lgrconvnet3": fixed_models.GenerativeResnetNoDetach, "lggcnn": fixed_models.LGGCNNNoDetach}

    def get_network(name):
        return fixed.get(name.lower()) or original(name)

    get_network.__wrapped__ = original
    official.get_network = get_network


def clip_fp32_for_clipfusion() -> None:
    """Cast CLIP-Fusion's CLIP to fp32 right after the official constructor.

    ``clip.load`` keeps fp16 weights on GPU (CLIP @ a1d0717, ``clip.py:139``), and ``inference/models/clipfusion.py``
    leaves them trainable, so the Adam step of ``train_network.py:307`` turns them into inf/NaN after one step (eps=1e-8
    is 0 in fp16; verified in our env, ``experiments/diag_clipfusion_nan.py``). Everything else is unchanged.
    """
    refs.add_lgd_to_path()
    import inference.models as official

    original = getattr(official.get_network, "__wrapped__", official.get_network)

    def get_network(name):
        cls = original(name)
        if name.lower() != "clipfusion":
            return cls

        def build(*args, **kwargs):
            net = cls(*args, **kwargs)
            net.clip_model.float()
            return net

        return build

    get_network.__wrapped__ = original
    official.get_network = get_network
