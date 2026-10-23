"""Student-t intervals use the clips that enter the published mean."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from latentcodecbench.analysis.intervals import (
    classic_bd_rate_samples,
    student_t_interval,
)
from latentcodecbench.analysis.plot import mean_bd_rate
from latentcodecbench.analysis.published import load_published
from latentcodecbench.analysis.report_tables import (
    interval_matches_mean,
    recon_interval_markdown,
    report_numbers,
)

ROOT = Path(__file__).resolve().parents[1]


def test_three_points_match_the_hand_calculation() -> None:
    interval = student_t_interval([1.0, 2.0, 3.0])
    half = 4.302652729749462 / math.sqrt(3)
    assert interval.n == 3
    assert interval.mean == pytest.approx(2.0)
    assert interval.low == pytest.approx(2.0 - half)
    assert interval.high == pytest.approx(2.0 + half)
    assert interval_matches_mean([1.0, 2.0, 3.0])


def test_one_point_has_no_interval() -> None:
    interval = student_t_interval([5.0])
    assert interval.n == 1
    assert interval.mean == pytest.approx(5.0)
    assert math.isnan(interval.low)
    assert math.isnan(interval.high)
    with pytest.raises(ValueError, match="no values"):
        student_t_interval([float("nan")])


def test_classic_samples_are_the_mean_the_readme_uses() -> None:
    tables = load_published(ROOT)
    for metric in ("psnr_y", "vmaf"):
        samples = classic_bd_rate_samples(tables.classic, metric=metric)
        for summary in mean_bd_rate(tables.classic, metric=metric):
            values = samples[summary.codec]
            assert len(values) == summary.clips
            assert student_t_interval(values).mean == pytest.approx(summary.mean_bd_rate)


def test_report_contains_the_rebuilt_intervals() -> None:
    tables = load_published(ROOT)
    numbers = report_numbers(tables.classic, tables.recon, tables.bitrate, tables.propagation)
    assert numbers in (ROOT / "docs" / "REPORT.md").read_text(encoding="utf-8")
    recon = recon_interval_markdown(tables.recon)
    note = ROOT / "docs" / "learn" / "13-confidence-intervals.md"
    assert recon in note.read_text(encoding="utf-8")
    assert recon in numbers
