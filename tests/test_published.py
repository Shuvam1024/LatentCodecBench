"""The published pages stay equal to the tables rebuilt from the CSVs."""

from __future__ import annotations

from pathlib import Path

from latentcodecbench.analysis.published import check_published, headline_markdown, load_published
from latentcodecbench.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_headline_matches_the_known_means() -> None:
    text = headline_markdown(load_published(ROOT))
    assert "| x265 | x264 | PSNR-Y | -24.66% | 34 | 85.37% | 0 |" in text
    assert "| wan2_1 | x264 | PSNR-Y | +824.24% | 33 | 32.27% | 33 |" in text
    assert "| ltx | 34 | 32.30 dB | 74.56 |" in text
    assert text in (ROOT / "README.md").read_text(encoding="utf-8")
    note = ROOT / "docs" / "learn" / "12-reproducing-the-tables.md"
    assert text in note.read_text(encoding="utf-8")


def test_published_pages_match_and_the_command_passes() -> None:
    assert check_published(ROOT) == []
    assert main(["reproduce", "--root", str(ROOT)]) == 0


def test_a_root_without_the_csvs_is_reported(tmp_path: Path) -> None:
    problems = check_published(tmp_path)
    assert problems
    assert main(["reproduce", "--root", str(tmp_path)]) == 1
