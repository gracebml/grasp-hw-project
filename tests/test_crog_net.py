import importlib
import sys

import pytest

pytest.importorskip("torch")
from lgdbench import crog_net  # noqa: E402


def test_crog_on_path_swaps_and_restores_utils(tmp_path, monkeypatch):
    # LGD-like utils already imported, CROG-like tree with its own utils that its model package imports
    lgd = tmp_path / "lgd" / "utils"
    lgd.mkdir(parents=True)
    (lgd / "__init__.py").write_text("WHO = 'lgd'\n")
    crog = tmp_path / "crog"
    (crog / "utils").mkdir(parents=True)
    (crog / "utils" / "__init__.py").write_text("WHO = 'crog'\n")
    (crog / "fakemodel").mkdir()
    (crog / "fakemodel" / "__init__.py").write_text("from utils import WHO\n")

    monkeypatch.syspath_prepend(str(tmp_path / "lgd"))
    for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        monkeypatch.delitem(sys.modules, k)
    lgd_utils = importlib.import_module("utils")
    assert lgd_utils.WHO == "lgd"

    monkeypatch.setattr(crog_net, "CROG", crog)
    with crog_net._crog_on_path():
        fakemodel = importlib.import_module("fakemodel")
    assert fakemodel.WHO == "crog"
    assert sys.modules["utils"] is lgd_utils
    assert str(crog) not in sys.path
    sys.modules.pop("fakemodel")
