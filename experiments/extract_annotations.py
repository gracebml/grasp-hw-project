"""Extract only the needed members of a Grasp-Anything(++) zip into the layout the official loader reads.

    python experiments/extract_annotations.py instructions <grasp_instructions.zip> <data_root>
    python experiments/extract_annotations.py masks <mask.zip> <data_root>

- instructions: one ``<scene>_<obj>_<part>.pkl`` per label file in
  ``<data_root>/grasp-anything++/positive_grasp`` -> ``<data_root>/grasp-anything++/grasp_instructions/``; each one must
  unpickle (no code) to a non-empty string.
- masks (CROG training only): ``<scene>_<obj>.npy`` for the keys of the official ``train/seen.obj``
  -> ``<data_root>/grasp-anything/mask/``. Validation and test never read masks.

The archives hold 1.9M-4.4M members, so the central directory is streamed (``ziputil``), never listed in memory, and
nothing else is unpacked. Files already present are kept (idempotent). Prints a JSON report. Instructions: exits 1
if a label file has no instruction, raises if one fails the check. Masks: missing keys are only reported (423 labelled
train keys have no mask in mask.zip).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lgdbench import data, refs  # noqa: E402
from lgdbench.ziputil import ZipReader, stem  # noqa: E402


def extract(zip_path, wanted: set[str], ext: str, dest: Path, check=None) -> dict:
    """Write member ``<stem><ext>`` of ``zip_path`` to ``dest/<stem><ext>`` for every stem in ``wanted``.
    ``check(path)`` validates each written file (raises on failure)."""
    dest.mkdir(parents=True, exist_ok=True)
    zr = ZipReader(zip_path)
    found, written, kept, duplicates = set(), 0, 0, 0
    for name, lho, csize, method in zr.entries():
        st = stem(name)
        if not name.endswith(ext) or st not in wanted:
            continue
        if st in found:
            duplicates += 1
            continue
        found.add(st)
        out = dest / f"{st}{ext}"
        if out.exists():
            kept += 1
        else:
            tmp = out.with_suffix(out.suffix + ".part")
            tmp.write_bytes(zr.read(lho, csize, method))
            tmp.rename(out)
            written += 1
        if check is not None:
            check(out)
    missing = sorted(wanted - found)
    return {"wanted": len(wanted), "found": len(found), "written": written, "already_present": kept,
            "duplicates_skipped": duplicates, "missing": len(missing), "missing_examples": missing[:10]}


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("instructions", "masks"):
        sys.exit(__doc__)
    kind, zip_path, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if kind == "instructions":
        labels = root / "grasp-anything++" / "positive_grasp"
        wanted = {p.name[:-3] for p in labels.iterdir() if p.name.endswith(".pt")}
        report = extract(zip_path, wanted, ".pkl", root / "grasp-anything++" / "grasp_instructions",
                         check=data.read_instruction)
    else:
        wanted = set(data.read_keys(refs.LGD / "split" / "grasp-anything++" / "train" / "seen.obj"))
        report = extract(zip_path, wanted, ".npy", root / "grasp-anything" / "mask")
    print(json.dumps({"kind": kind, "zip": str(zip_path), **report}, indent=1))
    # masks: some train keys have no mask in mask.zip (those samples leave CROG's training sampler)
    if kind == "instructions" and report["missing"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
