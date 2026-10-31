"""Train one method with the official LGD ``train()`` and GPU augmentation.

Mirrors ``train_network.run()``: same 90/10 split without shuffling, logging format, checkpoint naming and saving rule
(train_network.py:345), width decode x150. The image rotate/zoom/resize runs on the GPU (``lgdbench.gpuaug``) and so does the
validation scoring (``lgdbench.gpuval``, checked equal to the official ``validate`` on all 2,863 validation samples).
Validation images are not augmented.

Parameters (env):
    NETWORK      lgrconvnet3 | lggcnn | crog                                                            (required)
    DATA_ROOT    folder with grasp-anything/ and grasp-anything++/ (README)                             (required)
    GRAT         0 baseline (default) | 1 GRAT on that network
    OT           1 with the OT loss (default) | 0 the ablation "w/o OT"; GRAT=1 only
    SOURCE       relevance (default) | constant, the pose channels of x0 (the ablation); GRAT=1 only
    LAMBDA       weight of the OT loss, default grat.LAMBDA of the network (experiments/calibrate.py)
    SPLIT_LIST   default train/seen.obj
    EPOCHS, BATCHES, BATCH_SIZE, VAL_BS, NUM_WORKERS, SEED   default 50, 1000, 8, 32, 3, 123
    OUT_DIR      logs (default outputs/local); CKPT_ROOT checkpoints (default OUT_DIR)
    CLIP_RN50    OpenAI RN50.pt, CROG only (default ~/.cache/clip/RN50.pt)
    TEXT         instruction (default, GA++ grasp instruction) | object (official object name)
    RESUME       1 (default) writes the training state after every epoch to <CKPT_ROOT>/resume/ and continues from it when the
                 same command is launched again; a finished run renames it to .complete. 0 = no state written or read
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lgdbench import crog_net, data, fixes, gpuaug, gpuval, grat, refs  # noqa: E402

DEFAULT_TRAIN = refs.LGD / "split" / "grasp-anything++" / "train" / "seen.obj"


def env(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        sys.exit(f"missing required parameter {name}=...")
    return v


def resume_state(net, optimizer, scheduler, epoch, best_iou, save_folder, ckpt_folder):
    """Training state after ``epoch``: trainable parameters and buffers (frozen parameters such as CLIP are rebuilt by the
    constructor), optimizer, scheduler, RNGs and the run's folders."""
    import numpy as np
    import torch
    frozen = {n for n, q in net.named_parameters() if not q.requires_grad}
    return {"epoch": epoch + 1, "best_iou": best_iou, "save_folder": str(save_folder), "ckpt_folder": str(ckpt_folder),
            "model": {k: v for k, v in net.state_dict().items() if k not in frozen}, "frozen": sorted(frozen),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()}}


def write_resume(state, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, str(tmp))
    os.replace(tmp, path)


def load_resume(state, net, optimizer, scheduler):
    import numpy as np
    import torch
    missing, unexpected = net.load_state_dict(state["model"], strict=False)
    if unexpected or set(missing) - set(state["frozen"]):
        sys.exit(f"resume state does not match the network: missing {sorted(set(missing) - set(state['frozen']))[:5]}, "
                 f"unexpected {unexpected[:5]}")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state_all(state["rng"]["cuda"])


def main():
    network = env("NETWORK")
    use_grat = env("GRAT", "0") == "1"
    use_ot = use_grat and env("OT", "1") == "1"
    cfg = None
    if use_grat:
        cfg = grat.Config(lam=float(env("LAMBDA", str(grat.LAMBDA[network]))) if use_ot else 0.0,
                          source=env("SOURCE", "relevance"))
    split_list = Path(env("SPLIT_LIST", str(DEFAULT_TRAIN))).resolve()
    data_root = Path(env("DATA_ROOT")).resolve()
    out = Path(env("OUT_DIR", "outputs/local")).resolve()
    seed = int(env("SEED", "123"))
    epochs, batches, batch_size = int(env("EPOCHS", "50")), int(env("BATCHES", "1000")), int(env("BATCH_SIZE", "8"))
    workers = int(env("NUM_WORKERS", "3"))
    is_crog = network == "crog"

    import numpy as np
    import torch
    import torch.utils.data
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    refs.add_lgd_to_path()
    fixes.no_detach_baselines()
    if is_crog:
        crog_net.import_crog()
    from inference.models import get_network
    from train_network import train  # official training loop

    run = f"grat_{network}" + ("" if use_ot else "_noot") + ("" if cfg.source == "relevance" else "_constsource") if use_grat else network
    desc = f"{run}_{split_list.stem}_s{seed}"
    ckpt_root = Path(env("CKPT_ROOT", str(out))).resolve()
    use_resume = env("RESUME", "1") == "1"
    resume_path = ckpt_root / "resume" / f"{desc}.pt"
    state = torch.load(str(resume_path), map_location="cpu") if use_resume and resume_path.exists() else None
    if state is not None:
        save_folder, ckpt_folder = Path(state["save_folder"]), Path(state["ckpt_folder"])
    else:
        save_folder = out / f"{datetime.datetime.now().strftime('%y%m%d_%H%M')}_{desc}"
        ckpt_folder = ckpt_root / save_folder.name
    save_folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, filename=str(save_folder / "log.log"),
                        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
    logging.getLogger('').addHandler(console)

    mask_dir = str(data_root / "grasp-anything" / "mask") if is_crog else None
    text = env("TEXT", "instruction")
    Dataset = data.make_gpuaug_dataset_class(split_list, mask_dir=mask_dir, text=text, other_object=use_ot)
    dataset = Dataset(str(data_root / "grasp-anything"), output_size=224, ds_rotate=0.0, random_rotate=True,
                      random_zoom=True, include_depth=0, include_rgb=1, seen=1,
                      add_file_path=str(data_root / "grasp-anything++"))
    split = int(np.floor(0.9 * dataset.length))
    indices = list(range(dataset.length))
    train_idx, val_idx = indices[:split], indices[split:]
    dataset.split_pools = (set(train_idx), set(val_idx))  # the other object's labels never cross training and validation
    # same samples as the official split, no rotation / zoom (as evaluate.py)
    val_dataset = Dataset(str(data_root / "grasp-anything"), output_size=224, ds_rotate=0.0, random_rotate=False,
                          random_zoom=False, include_depth=0, include_rgb=1, seen=1,
                          add_file_path=str(data_root / "grasp-anything++"))
    val_dataset.split_pools = dataset.split_pools
    n_train_all = len(train_idx)
    if is_crog:
        train_idx = [i for i in train_idx if dataset.has_object_mask(i)]
    device = torch.device("cuda")
    norm = "clip" if is_crog else "lgd"
    loader = lambda idx, bs, ds=dataset: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=bs, num_workers=workers, pin_memory=True, persistent_workers=True,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idx))
    train_data = gpuaug.GPUAugLoader(loader(train_idx, batch_size), device, norm)
    val_data = gpuaug.GPUAugLoader(loader(val_idx, int(env("VAL_BS", "32")), val_dataset), device, norm)
    logging.info('Dataset size %d, training %d (of %d), validation %d', dataset.length, len(train_idx), n_train_all,
                 len(val_idx))

    scheduler = None
    clip_rn50 = os.path.expanduser(env("CLIP_RN50", "~/.cache/clip/RN50.pt"))
    if use_grat:
        net, params, crog_cfg = grat.build(network, cfg, clip_rn50)
        net = net.to(device)
        if is_crog:  # the CROG recipe on GRAT's parameter groups
            optimizer = torch.optim.Adam(params, lr=crog_cfg.base_lr, weight_decay=crog_cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=crog_cfg.milestones, gamma=crog_cfg.lr_decay)
        else:
            optimizer = torch.optim.Adam(params)  # train_network.py:307
    elif is_crog:
        net, param_list, crog_cfg = crog_net.build(clip_rn50)
        net = net.to(device)
        optimizer = torch.optim.Adam(param_list, lr=crog_cfg.base_lr, weight_decay=crog_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=crog_cfg.milestones, gamma=crog_cfg.lr_decay)
    else:
        net = get_network(network)(input_channels=3, dropout=1, prob=0.1, channel_size=32).to(device)
        optimizer = torch.optim.Adam(net.parameters())
    launch = {"network": network, "split_list": str(split_list), "n_train": len(train_idx), "n_val": len(val_idx),
              "epochs": epochs, "batches_per_epoch": batches, "batch_size": batch_size, "seed": seed,
              "num_workers": workers, "params": sum(p.numel() for p in net.parameters()), "decode_width": 150,
              "augment": "gpu", "validation": "gpu scoring, no augmentation", "val_bs": int(env("VAL_BS", "32")), "ckpt_folder": str(ckpt_folder), "text": text, "lgd_commit": refs.LGD_COMMIT,
              "grat": cfg.__dict__ if use_grat else None, "run": run}
    if is_crog:
        launch["crog_commit"] = crog_net.CROG_COMMIT
    launch["resume"] = {"enabled": use_resume, "path": str(resume_path),
                        "resumed_from_epoch": state["epoch"] if state is not None else None}
    (out / "launch.json").write_text(json.dumps(launch, indent=1))

    best_iou, start = 0.0, 0
    if state is not None:
        load_resume(state, net, optimizer, scheduler)
        best_iou, start = state["best_iou"], state["epoch"]
        logging.info('Resumed from epoch %d (best val %f), state %s', start, best_iou, resume_path)
        del state
    for epoch in range(start, epochs):
        logging.info('Beginning Epoch {:02d}, lr {}'.format(epoch, [g['lr'] for g in optimizer.param_groups]))
        train(epoch, net, device, train_data, optimizer, batches)
        logging.info('Validating...')
        res = gpuval.validate(net, device, val_data, 0.25)
        n = res['correct'] + res['failed']
        iou = res['correct'] / n
        logging.info('%d/%d = %f' % (res['correct'], n, iou))
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:  # train_network.py:345
            torch.save(net, str(ckpt_folder / ('epoch_%02d_iou_%0.2f' % (epoch, iou))))
            best_iou = iou
        if scheduler is not None:
            scheduler.step(epoch + 1)  # CROG train_crog.py:270
        if use_resume:
            write_resume(resume_state(net, optimizer, scheduler, epoch, best_iou, save_folder, ckpt_folder), resume_path)
    if use_resume and resume_path.exists():
        os.replace(resume_path, resume_path.with_suffix(".complete"))


if __name__ == "__main__":
    main()
