from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lgdbench import refs  # noqa: E402
from lgdbench.data import object_key, read_keys  # noqa: E402

SPLITS = {"train": "train/seen.obj", "test_seen": "test/seen.obj", "test_unseen": "test/unseen.obj"}
OFFICIAL_IMAGE_RE = re.compile(r"_\d{1}_\d{1}\.pt$")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def main(full: Path, subset: Path):
    keys = {s: set(read_keys(refs.LGD / "split" / "grasp-anything++" / p)) for s, p in SPLITS.items()}
    split_of = {k: s for s, ks in keys.items() for k in ks}

    files = defaultdict(list)  # object key -> grasp file names
    with os.scandir(full / "grasp-anything++" / "positive_grasp") as it:
        for e in it:
            if e.name.endswith(".pt") and object_key(e.name) in split_of:
                files[object_key(e.name)].append(e.name)

    report = {}
    for s, ks in keys.items():
        with_files = [k for k in ks if files.get(k)]
        names = [n for k in with_files for n in files[k]]
        scenes = {k.split("_")[0] for k in with_files}
        report[s] = {
            "keys": len(ks), "keys_with_grasp_files": len(with_files), "grasp_files": len(names),
            "scenes": len(scenes),
            "missing_image": sorted(x for x in scenes if not (full / "grasp-anything" / "image" / f"{x}.jpg").exists()),
            "missing_prompt": sorted(x for x in scenes if not (full / "grasp-anything" / "prompt" / f"{x}.pkl").exists()),
            "unmappable_by_official_regex": sorted(n for n in names if not OFFICIAL_IMAGE_RE.search(n)),
        }
        for n in names:
            link(full / "grasp-anything++" / "positive_grasp" / n, subset / "grasp-anything++" / "positive_grasp" / n)
        for x in scenes:
            for d, ext in (("image", "jpg"), ("prompt", "pkl")):
                src = full / "grasp-anything" / d / f"{x}.{ext}"
                if src.exists():
                    link(src, subset / "grasp-anything" / d / f"{x}.{ext}")

    (subset / "subset_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({s: {k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()}
                      for s, r in report.items()}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
