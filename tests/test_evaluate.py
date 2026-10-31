import importlib.util
import logging
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "evaluate.py"


@pytest.fixture
def evaluate():
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_captures_official_iou_line(evaluate, tmp_path):
    # mimics evaluate.py: basicConfig(level=INFO) at import, result logged with logging.info (evaluate.py:14, :170)
    fake = tmp_path / "evaluate.py"
    fake.write_text("import logging\nlogging.basicConfig(level=logging.INFO)\n"
                    "logging.info('IOU Results: %d/%d = %f' % (3, 10, 0.3))\n")
    root = logging.getLogger()
    old_level, old_handlers = root.level, list(root.handlers)
    root.setLevel(logging.WARNING)
    root.addHandler(logging.NullHandler())  # root already configured: the case that dropped every INFO line
    try:
        lines = evaluate.run_and_capture(fake)
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
    hits = [m for line in lines if (m := evaluate.IOU_LINE.search(line))]
    assert len(hits) == 1
    assert (int(hits[0][1]), int(hits[0][2])) == (3, 10)


def ckpts(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / name).write_text("")
    return d


def test_pick_checkpoint_file_names_without_log(evaluate, tmp_path):
    d = ckpts(tmp_path / "run", "epoch_00_iou_0.05", "epoch_04_iou_0.15", "epoch_10_iou_0.11", "log.txt")
    assert evaluate.pick_checkpoint(d).name == "epoch_04_iou_0.15"
    ckpts(d, "epoch_15_iou_0.15")  # tie at the top of the 2-digit names and no log: refuse
    with pytest.raises(SystemExit):
        evaluate.pick_checkpoint(d)


def test_pick_checkpoint_uses_full_precision_log(evaluate, tmp_path):
    # the CROG full run: ep38 814/2863 = 0.2843 and ep48 809/2863 = 0.2826 are both saved as iou_0.28
    d = ckpts(tmp_path / "run", "epoch_10_iou_0.21", "epoch_38_iou_0.28", "epoch_48_iou_0.28")
    log = tmp_path / "log.log"
    log.write_text("".join(f"x INFO - Beginning Epoch {e:02d}, lr [1e-05]\nx INFO - Validating...\nx INFO - {c}/2863 = 0.0\n"
                           for e, c in ((10, 608), (38, 814), (47, 700), (48, 809))))
    assert evaluate.val_ious(log)[38] == (814, 2863)
    assert evaluate.pick_checkpoint(d, log).name == "epoch_38_iou_0.28"
    log.write_text(log.read_text() + "x INFO - Beginning Epoch 49, lr [1e-05]\nx INFO - 814/2863 = 0.284317\n")
    ckpts(d, "epoch_49_iou_0.28")
    assert evaluate.pick_checkpoint(d, log).name == "epoch_49_iou_0.28"  # equal full IoU -> latest epoch


@pytest.mark.parametrize("tag,expected", [("instr_crog", "epoch_38_"),
                                          ("instr_explore_lggcnn_s125", "epoch_08_"),  # the re-run (ep 8 both times)
                                          ("instr_lgrconvnet3", "epoch_23_")])
def test_pick_checkpoint_on_the_real_logs(evaluate, tmp_path, tag, expected):
    logs = sorted((Path(__file__).resolve().parents[1] / "outputs").glob(f"train_{tag}/*/log.log"))
    if not logs:
        pytest.skip("run logs not pulled (outputs/ is not in git)")
    log = logs[-1]
    names = [f"epoch_{e:02d}_iou_{c / n:0.2f}" for e, (c, n) in evaluate.val_ious(log).items()]
    assert evaluate.pick_checkpoint(ckpts(tmp_path / "run", *names), log).name.startswith(expected)
