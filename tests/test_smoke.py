"""Import and command-line smoke tests."""

from __future__ import annotations

import subprocess

import pytest

import latentcodecbench
from latentcodecbench.cli import main


def test_version_is_set() -> None:
    assert latentcodecbench.__version__ == "0.0.1"


def test_phase_packages_import() -> None:
    import latentcodecbench.analysis
    import latentcodecbench.codecs
    import latentcodecbench.data
    import latentcodecbench.metrics
    import latentcodecbench.postfilter
    import latentcodecbench.vaes

    assert latentcodecbench.data.__doc__
    assert latentcodecbench.metrics.__doc__
    assert latentcodecbench.codecs.__doc__
    assert latentcodecbench.vaes.__doc__
    assert latentcodecbench.analysis.__doc__
    assert latentcodecbench.postfilter.__doc__


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"latentcodecbench {latentcodecbench.__version__}"


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage: lcb" in captured.out


def test_installed_script_reports_version() -> None:
    completed = subprocess.run(
        ["lcb", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"latentcodecbench {latentcodecbench.__version__}"
