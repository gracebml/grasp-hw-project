from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lgdbench import refs  # noqa: E402
from lgdbench.data import _NoGlobals, read_keys  # noqa: E402
from lgdbench.ziputil import ZipReader, stem  # noqa: E402

SEED = 0
UNSEEN_NAME_FRACTION = 0.30
TARGET_PROMPTS = {"train": 4000, "seen": 500, "unseen": 500}


def load_inputs(raw: Path):
    train_keys = set(read_keys(refs.LGD / "split" / "grasp-anything++" / "train" / "seen.obj"))

    prompts = defaultdict(int)  # <scene>_<obj> -> number of part-level grasp files (= prompts)
    for name, *_ in ZipReader(raw / "grasp-anything-pp" / "grasp_label_positive.zip").entries():
        if name.endswith(".pt"):
            key = stem(name).rsplit("_", 1)[0]
            if key in train_keys:
                prompts[key] += 1

    scenes = {k.split("_")[0] for k in train_keys}
    zr = ZipReader(raw / "grasp-anything" / "scene_description.zip")
    objects = {}
    for name, *rec in zr.entries():
        s = stem(name)
        if s in scenes:
            _, objects[s] = _NoGlobals(io.BytesIO(zr.read(*rec))).load()

    names = {}
    for k in train_keys:
        scene, obj = k.split("_")
        lst = objects.get(scene)
        if lst is not None and int(obj) < len(lst) and prompts.get(k):
            names[k] = lst[int(obj)]
    return train_keys, prompts, names


def take(keys, prompts, target):
    out, n = [], 0
    for k in keys:
        if n >= target:
            break
        out.append(k)
        n += prompts[k]
    return out


def cut(prompts, names, seed=SEED):
    rng = random.Random(seed)
    all_names = sorted(set(names.values()))
    unseen_names = set(rng.sample(all_names, round(len(all_names) * UNSEEN_NAME_FRACTION)))

    pool_unseen = sorted(k for k, n in names.items() if n in unseen_names)
    rng.shuffle(pool_unseen)
    unseen = take(pool_unseen, prompts, TARGET_PROMPTS["unseen"])
    unseen_scenes = {k.split("_")[0] for k in unseen}

    # Scene-disjoint: no image of explore_unseen appears in explore_train / explore_seen.
    by_scene = defaultdict(list)
    for k, n in names.items():
        s = k.split("_")[0]
        if n not in unseen_names and s not in unseen_scenes:
            by_scene[s].append(k)
    scene_order = sorted(by_scene)
    rng.shuffle(scene_order)

    train, n_train, i = [], 0, 0
    while n_train < TARGET_PROMPTS["train"] and i < len(scene_order):
        for k in sorted(by_scene[scene_order[i]]):
            train.append(k)
            n_train += prompts[k]
        i += 1
    train_names = {names[k] for k in train}

    seen, n_seen = [], 0
    for s in scene_order[i:]:
        if n_seen >= TARGET_PROMPTS["seen"]:
            break
        for k in sorted(by_scene[s]):
            if names[k] in train_names:  # "seen" = object name present in explore_train
                seen.append(k)
                n_seen += prompts[k]
    return {"train": sorted(train), "seen": sorted(seen), "unseen": sorted(unseen)}, sorted(unseen_names)


def main():
    raw = Path(os.environ.get("RAW", "data/raw"))
    out = Path(os.environ.get("OUT", "data_index"))
    out.mkdir(parents=True, exist_ok=True)
    train_keys, prompts, names = load_inputs(raw)
    parts, unseen_names = cut(prompts, names)

    sums = []
    for p, keys in parts.items():
        f = out / f"explore_{p}.txt"
        f.write_text("\n".join(keys) + "\n")
        sums.append((f.name, hashlib.sha256(f.read_bytes()).hexdigest()))
    meta = {
        "source_split": "LGD split/grasp-anything++/train/seen.obj",
        "lgd_commit": refs.LGD_COMMIT,
        "seed": SEED,
        "unseen_name_fraction": UNSEEN_NAME_FRACTION,
        "target_prompts": TARGET_PROMPTS,
        "train_keys_total": len(train_keys),
        "train_keys_with_name_and_prompts": len(names),
        "object_names_total": len(set(names.values())),
        "unseen_names": unseen_names,
        "counts": {p: {"objects": len(k), "prompts": sum(prompts[x] for x in k),
                       "scenes": len({x.split('_')[0] for x in k}), "names": len({names[x] for x in k})}
                   for p, k in parts.items()},
    }
    (out / "explore_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    sums.append(("explore_meta.json", hashlib.sha256((out / "explore_meta.json").read_bytes()).hexdigest()))
    (out / "CHECKSUMS.tsv").write_text("".join(f"{n}\t{h}\n" for n, h in sums))
    print(json.dumps(meta["counts"], indent=1))


if __name__ == "__main__":
    main()
