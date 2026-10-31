import hashlib
import json
import pickle
from pathlib import Path

import pytest

from lgdbench import refs
from lgdbench.data import object_key, read_keys, select_grasp_files

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data_index"
RAW = ROOT / "data" / "raw"
TRAIN_SPLIT = refs.LGD / "split" / "grasp-anything++" / "train" / "seen.obj"


def test_object_key_matches_official_slicing():
    # grasp_anywhere_data.py:38 keeps x.split('/')[-1][:-5]
    assert object_key("/d/positive_grasp/abc123_0_2.pt") == "abc123_0"


def test_read_keys_obj_and_txt(tmp_path):
    (tmp_path / "s.obj").write_bytes(pickle.dumps(["a_0", "b_1"]))
    (tmp_path / "s.txt").write_text("a_0\n\nb_1\n")
    assert read_keys(tmp_path / "s.obj") == ["a_0", "b_1"]
    assert read_keys(tmp_path / "s.txt") == ["a_0", "b_1"]


def test_read_keys_refuses_code(tmp_path):
    (tmp_path / "evil.obj").write_bytes(pickle.dumps(Path("x")))
    with pytest.raises(pickle.UnpicklingError):
        read_keys(tmp_path / "evil.obj")


def test_select_grasp_files(tmp_path):
    d = tmp_path / "positive_grasp"
    d.mkdir()
    for n in ("s1_0_0", "s1_0_1", "s1_1_0", "s2_0_0"):
        (d / f"{n}.pt").touch()
    got = [Path(f).name for f in select_grasp_files(str(tmp_path), ["s1_0", "s2_0"])]
    assert got == ["s1_0_0.pt", "s1_0_1.pt", "s2_0_0.pt"]


# ---------------------------------------------------------------- explore index (protocol §4.1)
PARTS = ("train", "seen", "unseen")


def _parts():
    return {p: read_keys(INDEX / f"explore_{p}.txt") for p in PARTS}


def test_explore_checksums():
    for line in (INDEX / "CHECKSUMS.tsv").read_text().splitlines():
        name, digest = line.split("\t")
        assert hashlib.sha256((INDEX / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.skipif(not TRAIN_SPLIT.exists(), reason="LGD reference code not found")
def test_explore_inside_train_and_disjoint():
    train = set(read_keys(TRAIN_SPLIT))
    parts = _parts()
    for keys in parts.values():
        assert set(keys) <= train  # never touches test/*
    scenes = {p: {k.split("_")[0] for k in keys} for p, keys in parts.items()}
    assert not scenes["train"] & scenes["seen"]
    assert not scenes["train"] & scenes["unseen"]
    assert not scenes["seen"] & scenes["unseen"]


def test_explore_sizes():
    meta = json.loads((INDEX / "explore_meta.json").read_text())
    for p in PARTS:
        target, got = meta["target_prompts"][p], meta["counts"][p]["prompts"]
        assert abs(got - target) <= 0.1 * target, (p, got, target)


@pytest.mark.skipif(not (RAW / "grasp-anything" / "scene_description.zip").exists(), reason="raw zips not found")
def test_explore_names():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bei", ROOT / "experiments" / "build_explore_index.py")
    bei = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bei)
    _, _, names = bei.load_inputs(RAW)
    parts = _parts()
    meta = json.loads((INDEX / "explore_meta.json").read_text())
    unseen_names = set(meta["unseen_names"])
    train_names = {names[k] for k in parts["train"]}
    assert {names[k] for k in parts["unseen"]} <= unseen_names
    assert not train_names & unseen_names
    assert {names[k] for k in parts["seen"]} <= train_names


@pytest.mark.skipif(not TRAIN_SPLIT.exists(), reason="LGD reference code not found")
def test_make_split_subset(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mss", ROOT / "experiments" / "make_split_subset.py")
    mss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mss)
    k = sorted(read_keys(TRAIN_SPLIT))[0]
    scene = k.split("_")[0]
    full = tmp_path / "full"
    (full / "grasp-anything" / "image").mkdir(parents=True)
    (full / "grasp-anything" / "prompt").mkdir(parents=True)
    (full / "grasp-anything++" / "positive_grasp").mkdir(parents=True)
    (full / "grasp-anything" / "image" / f"{scene}.jpg").write_bytes(b"x")
    (full / "grasp-anything" / "prompt" / f"{scene}.pkl").write_bytes(b"x")
    for n in (f"{k}_0.pt", f"{k}_1.pt", "notinsplit_0_0.pt"):
        (full / "grasp-anything++" / "positive_grasp" / n).write_bytes(b"x")
    mss.main(full, tmp_path / "sub")
    got = sorted(p.name for p in (tmp_path / "sub" / "grasp-anything++" / "positive_grasp").iterdir())
    assert got == [f"{k}_0.pt", f"{k}_1.pt"]
    assert (tmp_path / "sub" / "grasp-anything" / "image" / f"{scene}.jpg").exists()
    rep = json.loads((tmp_path / "sub" / "subset_report.json").read_text())
    assert rep["train"]["keys_with_grasp_files"] == 1 and rep["train"]["missing_image"] == []


# ---------------------------------------------------------------- text = GA++ grasp instruction
def test_instruction_file_is_the_official_path():
    from lgdbench.data import instruction_file
    # grasp_anywhere_data.py:124-128: basename, split('.')[0] + '.pkl', joined to instruction_dir
    assert instruction_file("/i", "/d/positive_grasp/abc123_0_2.pt") == "/i/abc123_0_2.pkl"


def test_read_instruction(tmp_path):
    from lgdbench.data import read_instruction
    (tmp_path / "ok.pkl").write_bytes(pickle.dumps("Lift toothbrush by its bristles."))
    (tmp_path / "empty.pkl").write_bytes(pickle.dumps("  "))
    (tmp_path / "evil.pkl").write_bytes(pickle.dumps(Path("x")))
    assert read_instruction(tmp_path / "ok.pkl") == "Lift toothbrush by its bristles."
    with pytest.raises(ValueError):
        read_instruction(tmp_path / "empty.pkl")
    with pytest.raises(pickle.UnpicklingError):
        read_instruction(tmp_path / "evil.pkl")


def _tiny_dataset(tmp_path):
    """Two parts of one object: same image, same object name, different instructions."""
    import numpy as np
    from PIL import Image
    ga, gapp = tmp_path / "ga", tmp_path / "gapp"
    for d in (ga / "image", ga / "prompt", gapp / "positive_grasp", gapp / "grasp_instructions"):
        d.mkdir(parents=True)
    Image.fromarray(np.zeros((416, 416, 3), np.uint8)).save(ga / "image" / "s1.jpg")
    (ga / "prompt" / "s1.pkl").write_bytes(pickle.dumps(("A toothbrush on a sink.", ["sink", "toothbrush"])))
    for part, ins in ((0, "Lift toothbrush by its bristles."), (2, "Grab toothbrush on its handle.")):
        (gapp / "positive_grasp" / f"s1_1_{part}.pt").touch()
        (gapp / "grasp_instructions" / f"s1_1_{part}.pkl").write_bytes(pickle.dumps(ins))
    # the official constructor first keeps only test/seen.obj files and raises if none is left
    # (grasp_anywhere_data.py:36-43, :51); one such file makes it pass, the key list below leaves it out
    (gapp / "positive_grasp" / f"{read_keys(refs.LGD / 'split/grasp-anything++/test/seen.obj')[0]}_0.pt").touch()
    split = tmp_path / "keys.txt"
    split.write_text("s1_1\n")
    return ga, gapp, split


@pytest.mark.parametrize("text, expected", [
    ("instruction", ["Lift toothbrush by its bristles.", "Grab toothbrush on its handle."]),
    ("object", ["toothbrush", "toothbrush"]),
])
def test_dataset_text_source(tmp_path, text, expected):
    pytest.importorskip("cv2")  # the official loader imports it; run on the cluster
    from lgdbench import data
    ga, gapp, split = _tiny_dataset(tmp_path)
    cls = data.make_dataset_class(split, text)
    ds = cls(str(ga), output_size=224, random_rotate=False, random_zoom=False, include_depth=0, include_rgb=1, seen=1,
             add_file_path=str(gapp))
    got = [ds.get_prompts(i) for i in range(ds.length)]
    assert [q for _, q in got] == expected
    assert all(p == "A toothbrush on a sink." for p, _ in got)  # scene description untouched


def test_text_must_be_known():
    from lgdbench import data
    with pytest.raises(ValueError):
        data.make_dataset_class("unused.txt", "caption")
