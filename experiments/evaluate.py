from __future__ import annotations

import json
import logging
import os
import re
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lgdbench import crog_net, data, fixes, refs  # noqa: E402

IOU_LINE = re.compile(r"IOU Results: (\d+)/(\d+) = ([0-9.]+)")
CKPT_NAME = re.compile(r"epoch_(\d+)_iou_([0-9.]+)$")
VAL_LINE = re.compile(r"Beginning Epoch (\d+)|INFO - (\d+)/(\d+) = [0-9.]+")
REPO = Path(__file__).resolve().parents[1]


def val_ious(log: Path) -> dict[int, tuple[int, int]]:
    """{epoch: (correct, total)} of the validation lines of a training log ('Beginning Epoch NN', then 'c/n = x')."""
    out, epoch = {}, None
    for m in VAL_LINE.finditer(log.read_text()):
        if m[1] is not None:
            epoch = int(m[1])
        elif epoch is not None:
            out[epoch] = (int(m[2]), int(m[3]))
    return out


def find_val_log(ckpt_dir: Path) -> Path | None:
    """VAL_LOG, else the run's own log.log."""
    if os.environ.get("VAL_LOG"):
        return Path(os.environ["VAL_LOG"])
    log = ckpt_dir / "log.log"
    return log if log.exists() else None


def pick_checkpoint(ckpt_dir: Path, val_log: Path | None = None) -> Path:
    """Checkpoint of the epoch with the highest val IoU in the training log; file-name IoU only without a log."""
    files = {int(m[1]): (float(m[2]), p) for p in ckpt_dir.iterdir() if (m := CKPT_NAME.search(p.name))}
    if not files:
        sys.exit(f"no epoch_XX_iou_Y checkpoint in {ckpt_dir}")
    log = val_log or find_val_log(ckpt_dir)
    if log is not None:
        ious = val_ious(log)
        cands = [(c / n, e) for e, (c, n) in ious.items() if e in files]
        if cands:
            best = max((c / n, e) for e, (c, n) in ious.items())
            if best[1] not in files:
                print(f"warning: best val epoch {best[1]} of {log} has no checkpoint in {ckpt_dir}", file=sys.stderr)
            return files[max(cands)[1]][1]
    top = max(iou for iou, _ in files.values())
    tied = sorted(e for e, (iou, _) in files.items() if iou == top)
    if len(tied) > 1:
        sys.exit(f"epochs {tied} tie at val IoU {top} in the file names of {ckpt_dir} and no training log: set VAL_LOG")
    return files[tied[0]][1]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run_and_capture(script: Path) -> list[str]:
    """Run an official script as __main__ and return the messages it logs.

    evaluate.py:14 calls logging.basicConfig(level=INFO), which does nothing once the root logger has a handler; so
    configure the root logger as that call would (stderr handler, INFO) before attaching the capture handler.
    """
    root = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    root.setLevel(logging.INFO)
    cap = _Capture()
    root.addHandler(cap)
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        root.removeHandler(cap)
    return cap.lines


def main():
    ckpt_dir = Path(os.environ["CKPT_DIR"]).resolve()
    val_log = find_val_log(ckpt_dir)
    ckpt = pick_checkpoint(ckpt_dir, val_log)
    split_list = Path(os.environ["SPLIT_LIST"]).resolve()
    data_root = Path(os.environ["DATA_ROOT"]).resolve()
    out = Path(os.environ.get("OUT_DIR", "outputs/local")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    crog = os.environ.get("MODEL", "lgd") == "crog"
    if crog:
        crog_net.import_crog()  # to unpickle CROG checkpoints
    from lgdbench import grat
    grat.accept_released_checkpoints()
    text = os.environ.get("TEXT", "instruction")
    data.register(split_list, crog=crog, text=text)
    sys.argv = ["evaluate.py",
                "--network", str(ckpt),
                "--dataset", data.CROG_DATASET_NAME if crog else data.DATASET_NAME,
                "--dataset-path", str(data_root / "grasp-anything"),
                "--add-file-path", str(data_root / "grasp-anything++"),
                "--use-depth", "0",
                "--iou-eval",
                "--split", "0"]  # the default 0.01 drops the first 1% of the test set
    with refs.lgd_cwd():
        lines = run_and_capture(refs.LGD / "evaluate.py")

    hits = [m for line in lines if (m := IOU_LINE.search(line))]
    if len(hits) != 1:
        sys.exit(f"expected one 'IOU Results' line from evaluate.py, got {len(hits)}")
    correct, total, sr = int(hits[0][1]), int(hits[0][2]), float(hits[0][3])
    res = {"checkpoint": str(ckpt), "val_log": None if val_log is None else str(val_log),
           "split_list": str(split_list), "correct": correct, "total": total,
           "success_rate": correct / total, "official_printed_sr": sr, "argv": sys.argv,
           "decode_width": 150, "text": text,
           "lgd_commit": refs.LGD_COMMIT}
    (out / f"eval_{split_list.stem}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
