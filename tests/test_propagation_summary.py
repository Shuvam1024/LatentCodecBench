"""Phase 3 summary: robust retention, the additive PSNR model, and the note."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from latentcodecbench.analysis.propagation import PropagationRow, read_propagation
from latentcodecbench.analysis.propagation_summary import (
    additive_psnr,
    degraded_source_mismatches,
    pooled_retention,
    spread,
    summarize,
    summary_markdown,
)
from latentcodecbench.analysis.table import read_results

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "artifact-propagation.csv"
RECON = ROOT / "docs" / "results" / "vae-recon.csv"
NOTE = ROOT / "docs" / "learn" / "10-artifact-propagation.md"


def _row(**changes: object) -> PropagationRow:
    base = PropagationRow(
        run_id="run",
        git_sha="abc",
        config_sha256="c" * 64,
        clip_id="alpha",
        vae="wan2_1",
        source_codec="libx264",
        source_crf=18,
        preset="medium",
        width=16,
        height=16,
        frames=4,
        latent_mse=1.0,
        latent_relative_l2=0.1,
        latent_cosine=0.99,
        psnr_recon_clean=30.0,
        psnr_recon_degraded=35.0,
        psnr_degraded_clean=40.0,
        blocking_clean=0.0,
        blocking_degraded=1.0,
        blocking_recon=0.5,
        blocking_retention=0.5,
        ringing_clean=1.0,
        ringing_degraded=2.0,
        ringing_recon=1.5,
        ringing_retention=0.5,
        clean_encode_seconds=0.1,
        encode_seconds=0.1,
        decode_seconds=0.1,
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_spread_ignores_one_huge_outlier() -> None:
    s = spread([0.1, 0.2, 0.3, 0.4, -1403.0])
    assert s.median == pytest.approx(0.2)
    assert s.q1 == pytest.approx(0.1)
    assert s.q3 == pytest.approx(0.3)
    assert s.iqr == pytest.approx(0.2)
    assert math.isnan(spread([]).median)


def test_pooled_retention_weights_clips_by_their_gap() -> None:
    # A clip with a 1e-5 gap would score -1000 on its own. Pooled, it barely counts.
    value = pooled_retention([0.0, 0.0], [1.0, 1e-5], [0.5, -0.01])
    assert value == pytest.approx((0.5 - 0.01) / (1.0 + 1e-5))
    assert math.isnan(pooled_retention([1.0], [1.0], [2.0]))


def test_additive_psnr_adds_mse() -> None:
    assert additive_psnr(30.0, 30.0) == pytest.approx(30.0 - 10 * math.log10(2))
    assert additive_psnr(30.0, 200.0) == pytest.approx(30.0, abs=1e-9)


def test_mismatch_finds_a_degraded_file_that_changed_between_vaes() -> None:
    rows = [
        _row(vae="wan2_1", psnr_degraded_clean=40.0),
        _row(vae="ltx", psnr_degraded_clean=40.0),
        _row(vae="wan2_1", source_codec="libsvtav1", source_crf=28, psnr_degraded_clean=40.0),
        _row(vae="ltx", source_codec="libsvtav1", source_crf=28, psnr_degraded_clean=40.5),
    ]
    found = degraded_source_mismatches(rows)
    assert found["libx264"] == (0, 0.0)
    assert found["libsvtav1"][0] == 1
    assert found["libsvtav1"][1] == pytest.approx(0.5)


def test_summarize_groups_and_uses_the_clean_baseline() -> None:
    rows = [_row(clip_id=name) for name in ("alpha", "beta", "gamma")]
    summary = summarize(rows)
    assert len(summary) == 1
    only = summary[0]
    assert only.clips == 3
    assert only.blocking_pooled == pytest.approx(0.5)
    assert only.clean_recon_psnr is None
    assert only.additive_gap_db is None


@pytest.mark.skipif(not RESULTS.is_file(), reason="results table not committed")
def test_note_table_matches_the_committed_csv() -> None:
    rows = read_propagation(RESULTS)
    assert len(rows) == 1224
    assert len({row.config_sha256 for row in rows}) == 1
    table = summary_markdown(summarize(rows, read_results(RECON)))
    assert table in NOTE.read_text(encoding="utf-8")
