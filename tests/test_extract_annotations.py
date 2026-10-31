import importlib.util
import pickle
import zipfile
from pathlib import Path

import pytest

from lgdbench import refs

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "extract_annotations.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("extract_annotations", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def archive(tmp_path):
    z = tmp_path / "grasp_instructions.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("grasp_instructions/s1_0_0.pkl", pickle.dumps("Lift toothbrush by its bristles."))
        zf.writestr("grasp_instructions/s1_0_1.pkl", pickle.dumps(""))
        zf.writestr("grasp_instructions/s2_1_0.txt", b"not a pickle")
    return z


def test_extract_only_wanted_and_idempotent(mod, archive, tmp_path):
    from lgdbench.data import read_instruction
    dest = tmp_path / "out"
    rep = mod.extract(archive, {"s1_0_0", "s9_0_0"}, ".pkl", dest, check=read_instruction)
    assert (rep["found"], rep["written"], rep["missing"], rep["missing_examples"]) == (1, 1, 1, ["s9_0_0"])
    assert [p.name for p in dest.iterdir()] == ["s1_0_0.pkl"]
    assert read_instruction(dest / "s1_0_0.pkl") == "Lift toothbrush by its bristles."
    rep = mod.extract(archive, {"s1_0_0"}, ".pkl", dest, check=read_instruction)
    assert (rep["written"], rep["already_present"]) == (0, 1)


def test_extract_rejects_empty_instruction(mod, archive, tmp_path):
    from lgdbench.data import read_instruction
    with pytest.raises(ValueError):
        mod.extract(archive, {"s1_0_1"}, ".pkl", tmp_path / "out", check=read_instruction)


def test_missing_reference_repo_explains_the_fix(tmp_path):
    empty = tmp_path / "LGD"
    empty.mkdir()  # what an uninitialised submodule looks like
    for missing in (empty, tmp_path / "absent"):
        with pytest.raises(FileNotFoundError, match="git submodule update --init"):
            refs.require(missing)
    (empty / "train_network.py").touch()
    assert refs.require(empty) == empty
