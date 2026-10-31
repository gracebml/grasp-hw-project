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

