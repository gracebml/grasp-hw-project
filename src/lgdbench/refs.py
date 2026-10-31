"""Single entry point for the official LGD code (read-only, never copied into ``src/``).

Every reused piece (dataset class, training loop, metric) is imported through here, so the place where the original
code enters the project is explicit.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Git submodules pinned at the commits below (.gitmodules); read-only, every change lives in src/lgdbench/.
REF_ROOT = REPO_ROOT / "ref_sources"

LGD = REF_ROOT / "LGD"
LGD_COMMIT = "501904dc5c30c8fad58d90f7e4f49b4095f11eb0"

# Importing from the reference repo must not write __pycache__ into it.
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(REPO_ROOT / ".pycache")


def require(path: Path) -> Path:
    """Fail early with the fix when a submodule was not checked out (clone without --recursive leaves an empty
    directory). Checks content, not ``.git``: the cluster copy is synced without it."""
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f"official code missing at {path}; run: git submodule update --init")
    return path


def add_lgd_to_path() -> Path:
    """Append (not prepend) LGD to ``sys.path``: its generic top-level names (``utils``, ``inference``) must not shadow
    installed packages."""
    require(LGD)
    p = str(LGD)
    if p not in sys.path:
        sys.path.append(p)
    return LGD


@contextmanager
def lgd_cwd():
    """The official code opens ``split/...`` relative to the working directory, so run it from the LGD root."""
    require(LGD)
    old = os.getcwd()
    os.chdir(LGD)
    try:
        yield LGD
    finally:
        os.chdir(old)
