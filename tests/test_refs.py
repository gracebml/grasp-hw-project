import subprocess

import pytest

from lgdbench import crog_net, refs


@pytest.mark.parametrize("path, commit", [(refs.LGD, refs.LGD_COMMIT), (crog_net.CROG, crog_net.CROG_COMMIT)])
def test_official_code_is_the_pinned_untouched_commit(path, commit):
    if not (path / ".git").exists():
        pytest.skip(f"{path} is not a git checkout (cluster copy or submodule not initialised)")
    git = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
    assert git("rev-parse", "HEAD").strip() == commit
    assert git("status", "--porcelain", "--untracked-files=no") == ""  # no edit to the official code
