"""The wheel records the commit it was built from."""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from latentcodecbench import _build_info
from latentcodecbench.analysis.run import git_sha

ROOT = Path(__file__).resolve().parents[1]


def test_checkout_build_info_is_empty() -> None:
    assert _build_info.GIT_SHA == ""
    text = (ROOT / "src" / "latentcodecbench" / "_build_info.py").read_text(encoding="utf-8")
    assert 'GIT_SHA = ""' in text


def test_git_sha_prefers_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "a" * 40)
    monkeypatch.setattr(_build_info, "GIT_SHA", "b" * 40)
    assert git_sha() == "a" * 40


def test_git_sha_reads_the_recorded_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LATENTCODECBENCH_GIT_SHA", raising=False)
    monkeypatch.setattr(_build_info, "GIT_SHA", "c" * 40)
    assert git_sha() == "c" * 40


def test_wheel_embeds_the_build_commit(tmp_path: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = ROOT / "src" / "latentcodecbench" / "_build_info.py"
    original = source.read_text(encoding="utf-8")
    dest = tmp_path / "wheels"
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dest)],
            cwd=ROOT,
            check=True,
            env=env,
        )
        wheels = sorted(dest.glob("*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as archive:
            recorded = archive.read("latentcodecbench/_build_info.py").decode()
        assert f'GIT_SHA = "{head}"' in recorded
    finally:
        source.write_text(original, encoding="utf-8")
    assert 'GIT_SHA = ""' in source.read_text(encoding="utf-8")
