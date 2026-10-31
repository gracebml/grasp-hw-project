"""Split and text adapter for the official LGD loader.

``PaperSplitDataset`` subclasses the official ``GraspAnywhereDataset`` and only replaces the list of grasp files it
iterates over and, with ``text="instruction"`` (default), the text it returns: the GA++ grasp instruction of the
part instead of the object name. Reading images, grasp labels and the scene description stays the official code.
The CROG variant additionally normalises the image with CLIP's mean/std and, for training only, appends the GA
object mask to the targets.
"""
from __future__ import annotations

import glob
import io
import os
import pickle
from pathlib import Path

from . import refs

DATASET_NAME = "grasp-anything++-paper"
TEXTS = ("instruction", "object")  # GA++ grasp instruction | official get_prompts
CROG_DATASET_NAME = "grasp-anything++-paper-clipnorm"
# CLIP normalisation as in CROG utils/dataset.py:125-127
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _NoGlobals(pickle.Unpickler):
    """The split pickles only hold a list of strings; refuse anything that would import code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def read_keys(path: str | os.PathLike) -> list[str]:
    """``<scene>_<obj>`` keys from an official ``.obj`` split or a ``data_index/*.txt`` list."""
    path = Path(path)
    if path.suffix == ".obj":
        return list(_NoGlobals(io.BytesIO(path.read_bytes())).load())
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def object_key(grasp_file: str) -> str:
    """``.../<scene>_<obj>_<part>.pt`` -> ``<scene>_<obj>``, the same slicing as the official filter
    (``grasp_anywhere_data.py:38``)."""
    return os.path.basename(grasp_file)[:-5]


def select_grasp_files(add_file_path: str, keys) -> list[str]:
    keys = set(keys)
    files = glob.glob(os.path.join(add_file_path, "positive_grasp", "*.pt"))
    return sorted(f for f in files if object_key(f) in keys)


def object_mask_file(mask_dir: str, grasp_file: str) -> str:
    """GA object mask of a grasp file: ``<mask_dir>/<scene>_<obj>.npy``."""
    return os.path.join(mask_dir, object_key(grasp_file) + ".npy")


def instruction_file(instruction_dir: str, grasp_file: str) -> str:
    """GA++ grasp instruction of a grasp file: ``<instruction_dir>/<scene>_<obj>_<part>.pkl``, the path the official
    ``get_prompts`` builds (``grasp_anywhere_data.py:126-128``) before its reading lines are commented out."""
    return os.path.join(instruction_dir, os.path.basename(grasp_file).split(".")[0] + ".pkl")


def read_instruction(path: str) -> str:
    text = _NoGlobals(io.BytesIO(Path(path).read_bytes())).load()
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path}: expected a non-empty instruction string, got {text!r}")
    return text


def make_dataset_class(split_file: str | os.PathLike, text: str = "instruction"):
    """Subclass of the official dataset bound to one key list; ``text`` picks the returned query."""
    if text not in TEXTS:
        raise ValueError(f"text must be one of {TEXTS}, got {text!r}")
    refs.add_lgd_to_path()
    from utils.data.grasp_anywhere_data import GraspAnywhereDataset

    keys = read_keys(split_file)

    class PaperSplitDataset(GraspAnywhereDataset):
        SPLIT_FILE = str(split_file)

        def __init__(self, file_path, ds_rotate=0, **kwargs):
            # The official constructor sets every path attribute; it reads split/... relative to the cwd.
            with refs.lgd_cwd():
                super().__init__(file_path, ds_rotate=ds_rotate, **kwargs)
            self.grasp_files = select_grasp_files(kwargs["add_file_path"], keys)
            self.length = len(self.grasp_files)
            if self.length == 0:
                raise FileNotFoundError(f"no grasp file of {split_file} under {kwargs['add_file_path']}")

        def get_prompts(self, idx):
            prompt, name = super().get_prompts(idx)  # official: (scene description, object name)
            if text == "object":
                return prompt, name
            return prompt, read_instruction(instruction_file(self.instruction_dir, self.grasp_files[idx]))

    return PaperSplitDataset


def make_crog_dataset_class(split_file: str | os.PathLike, mask_dir: str | None = None, text: str = "instruction"):
    """CROG variant of ``PaperSplitDataset``.

    - ``get_rgb`` returns the same pixels as the official one (same rotate/zoom/resize) normalised with CLIP's
      mean/std instead of "/255 minus the image mean";
    - with ``mask_dir`` (training only) the targets get a 5th entry, the GA object mask under the same rotate/zoom/resize
      as the image. Validation and test never pass ``mask_dir``. 423 labelled train keys have no mask in GA's mask.zip:
      they get a mask of -1 (``has_object_mask`` is False) and are left out of CROG's training sampler.
    """
    import numpy as np
    from utils.dataset_processing import image

    base = make_dataset_class(split_file, text)
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)

    class CROGSplitDataset(base):
        def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
            rgb = super().get_rgb(idx, rot, zoom, normalise=False)
            if not normalise:
                return rgb
            return ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose((2, 0, 1))

        def has_object_mask(self, idx):
            return os.path.exists(object_mask_file(mask_dir, self.grasp_files[idx]))

        def get_object_mask(self, idx, rot=0, zoom=1.0):
            if not self.has_object_mask(idx):
                return np.full((1, self.output_size, self.output_size), -1.0, dtype=np.float32)
            m = image.Image(np.load(object_mask_file(mask_dir, self.grasp_files[idx])).astype(np.float32))
            # the transforms of the official get_rgb (grasp_anywhere_data.py get_rgb), in the same order
            m.rotate(rot)
            m.zoom(zoom)
            m.resize((self.output_size, self.output_size))
            return (m.img > 0.5).astype(np.float32)[None]

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            if mask_dir is None:
                return item
            x, y, didx, rot, zoom = item[:5]
            mask = self.get_object_mask(didx, rot, zoom)
            return (x, (*y, self.numpy_to_torch(mask)), didx, rot, zoom, *item[5:])

    return CROGSplitDataset


def make_gpuaug_dataset_class(split_file: str | os.PathLike, mask_dir: str | None = None, text: str = "instruction",
                              other_object: bool = False):
    """``PaperSplitDataset`` for GPU augmentation.

    The official ``__getitem__`` still draws ``rot``/``zoom`` and builds the grasp targets with the official
    ``get_gtbb``/``draw``; only ``get_rgb`` returns the raw 416x416 uint8 image, which ``gpuaug.GPUAugLoader`` rotates, zooms,
    resizes and normalises on the GPU with the returned ``rot``/``zoom``. With ``mask_dir`` (CROG training) the raw
    GA object mask is appended to the targets (-1 everywhere when the key has no mask). With ``other_object`` (the OT
    loss of GRAT) the quality map of ANOTHER object of the image comes last (official ``get_gtbb`` and ``draw`` with the
    sample's ``rot``/``zoom``, different GT grasps, inside the sample's pool of ``split_pools``; zeros if there is none).
    """
    import hashlib
    import re

    import numpy as np
    import torch

    base = make_dataset_class(split_file, text)  # puts the LGD code on sys.path
    from utils.dataset_processing import image

    class GPUAugSplitDataset(base):
        def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
            # the file the official get_rgb reads (grasp_anywhere_data.py get_rgb), untouched: 416x416x3 uint8;
            # the official one always ends with resize(output_size), so it cannot be reused here
            rgb_file = re.sub(r"_\d{1}_\d{1}\.pt", ".jpg", self.grasp_files[idx]).split('/')[-1]
            return image.Image.from_file(os.path.join(self.rgb_dir, rgb_file)).img

        split_pools: tuple = ()

        def has_object_mask(self, idx):
            return os.path.exists(object_mask_file(mask_dir, self.grasp_files[idx]))

        def same_image(self, idx):
            """Indices of the files of the same image as ``idx`` within its pool (all files if no pool holds it)."""
            if not hasattr(self, "_by_image"):
                self._by_image = {}
                for i, f in enumerate(self.grasp_files):
                    self._by_image.setdefault(object_key(f).rsplit("_", 1)[0], []).append(i)
            group = self._by_image[object_key(self.grasp_files[idx]).rsplit("_", 1)[0]]
            pool = next((p for p in self.split_pools if idx in p), None)
            return [i for i in group if pool is None or i in pool]

        def label_digest(self, idx):
            if not hasattr(self, "_digest"):
                self._digest = {}
            if idx not in self._digest:
                with open(self.grasp_files[idx], "rb") as fh:
                    self._digest[idx] = hashlib.md5(fh.read()).hexdigest()
            return self._digest[idx]

        def partner(self, idx):
            """The next file of the same image within the pool (cyclic) that belongs to another object and whose GT grasps
            DIFFER from the sample's, or None."""
            group = self.same_image(idx)
            k = group.index(idx)
            own = object_key(self.grasp_files[idx])
            for j in group[k + 1:] + group[:k]:
                if object_key(self.grasp_files[j]) != own and self.label_digest(j) != self.label_digest(idx):
                    return j
            return None

        def __getitem__(self, idx):
            x, y, didx, rot, zoom, *rest = super().__getitem__(idx)
            x = x.to(torch.uint8)  # numpy_to_torch made it float32; values are the uint8 pixels
            if mask_dir is not None:
                if self.has_object_mask(didx):
                    m = torch.from_numpy(np.load(object_mask_file(mask_dir, self.grasp_files[didx])).astype(np.float32))
                else:
                    m = torch.full(x.shape[:2], -1.0)
                y = (*y, m[None])  # 5th target
            if other_object:
                j = self.partner(didx)
                size = (self.output_size, self.output_size)
                pmap = self.get_gtbb(j, rot, zoom).draw(size)[0].astype(np.float32) if j is not None else np.zeros(size, np.float32)
                y = (*y, self.numpy_to_torch(pmap))
            return (x, y, didx, rot, zoom, *rest)

    return GPUAugSplitDataset


def register(split_file: str | os.PathLike, crog: bool = False, text: str = "instruction") -> None:
    """Make ``get_dataset(DATASET_NAME)`` (or ``CROG_DATASET_NAME`` with ``crog``) return the split-bound subclass;
    every other name goes to the official function. Must run before the official script does
    ``from utils.data import get_dataset``."""
    refs.add_lgd_to_path()
    import utils.data as official

    if not Path(official.__file__).resolve().is_relative_to(refs.LGD.resolve()):
        raise ImportError(f"utils.data resolved to {official.__file__}, not the LGD reference code")
    original = getattr(official.get_dataset, "__wrapped__", official.get_dataset)
    cls = make_crog_dataset_class(split_file, text=text) if crog else make_dataset_class(split_file, text)
    name = CROG_DATASET_NAME if crog else DATASET_NAME

    def get_dataset(n):
        return cls if n == name else original(n)

    get_dataset.__wrapped__ = original
    official.get_dataset = get_dataset
