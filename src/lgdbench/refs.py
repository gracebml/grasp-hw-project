from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_ROOT = REPO_ROOT / "ref_sources"

LGD = REF_ROOT / "LGD"
LGD_COMMIT = "501904dc5c30c8fad58d90f7e4f49b4095f11eb0"

# Importing from the reference repo must not write __pycache__ into it.
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(REPO_ROOT / ".pycache")


def require(path: Path) -> Path:
    """Fail early when the submodule was not checked out."""
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f"official code missing at {path}; run: git submodule update --init")
    return path


def add_lgd_to_path() -> Path:
    """Append, not prepend: LGD's top-level names (``utils``, ``inference``) must not shadow installed packages."""
    require(LGD)
    p = str(LGD)
    if p not in sys.path:
        sys.path.append(p)
    return LGD


@contextmanager
def lgd_cwd():
    """Run from the LGD root: the official code opens ``split/...`` relative to the working directory."""
    require(LGD)
    old = os.getcwd()
    os.chdir(LGD)
    try:
        yield LGD
    finally:
        os.chdir(old)
