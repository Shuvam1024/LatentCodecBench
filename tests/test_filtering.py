"""The training-data filter is the propagation rows minus the clean reconstruction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from latentcodecbench.analysis.filtering import (
    filter_points,
    filtering_report,
    rising_latent_curves,
)
from latentcodecbench.analysis.propagation import PropagationRow, read_propagation
from latentcodecbench.analysis.table import ResultRow, read_results

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "artifact-propagation.csv"
RECON = ROOT / "docs" / "results" / "vae-recon.csv"
NOTE = ROOT / "docs" / "learn" / "11-training-data-filter.md"


def _propagation(**changes: object) -> PropagationRow:
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
        latent_relative_l2=0.2,
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


def _recon(clip_id: str, codec: str, psnr: float) -> ResultRow:
    return ResultRow(
        run_id="run",
        git_sha="a" * 40,
        config_sha256="b" * 64,
        clip_id=clip_id,
        codec=codec,
        preset="medium",
        crf=None,
        width=16,
        height=16,
        frames=4,
        fps_num=25,
        fps_den=1,
        bits=None,
        bpp=None,
        kbps=None,
        encode_seconds=0.1,
        decode_seconds=0.1,
        psnr_y=psnr,
        psnr_rgb=psnr,
        ssim=0.9,
        vmaf=80.0,
        flicker=0.1,
        lpips=None,
    )


def test_delta_is_degraded_reconstruction_minus_clean_reconstruction() -> None:
    rows = [
        _propagation(clip_id="alpha", psnr_recon_clean=31.0, latent_relative_l2=0.1),
        _propagation(clip_id="beta", psnr_recon_clean=29.0, latent_relative_l2=0.2),
        _propagation(clip_id="gamma", psnr_recon_clean=30.0, latent_relative_l2=0.3),
    ]
    recon = [
        _recon("alpha", "wan2_1", 30.0),
        _recon("beta", "wan2_1", 30.0),
        _recon("gamma", "wan2_1", 30.0),
    ]
    point = filter_points(rows, recon)[0]
    assert point.clips == 3
    assert point.delta_median == pytest.approx(0.0)
    assert point.delta_q1 == pytest.approx(-0.5)
    assert point.delta_q3 == pytest.approx(0.5)
    assert point.clips_above_clean == 1
    assert point.latent_relative_l2_median == pytest.approx(0.2)


def test_missing_or_duplicate_clean_reconstruction_is_an_error() -> None:
    row = _propagation()
    with pytest.raises(ValueError, match="no clean reconstruction"):
        filter_points([row], [])
    recon = [_recon("alpha", "wan2_1", 30.0), _recon("alpha", "wan2_1", 31.0)]
    with pytest.raises(ValueError, match="duplicate clean reconstruction"):
        filter_points([row], recon)


def test_a_curve_rises_only_when_every_step_does() -> None:
    flat = [
        _propagation(source_crf=18, latent_relative_l2=0.1),
        _propagation(source_crf=30, latent_relative_l2=0.1),
    ]
    rising = [
        _propagation(clip_id="beta", source_crf=18, latent_relative_l2=0.1),
        _propagation(clip_id="beta", source_crf=30, latent_relative_l2=0.2),
    ]
    assert rising_latent_curves(flat + rising) == (1, 2)


@pytest.mark.skipif(not RESULTS.is_file(), reason="results table not committed")
def test_note_matches_the_committed_csvs() -> None:
    rows = read_propagation(RESULTS)
    recon = read_results(RECON)
    report = filtering_report(rows, recon)
    assert report in NOTE.read_text(encoding="utf-8")
    assert rising_latent_curves(rows) == (408, 408)
    points = filter_points(rows, recon)
    assert {point.clips for point in points} == {34}
    harsh = [
        point
        for point in points
        if point.source_crf == {"libx264": 38, "libx265": 38, "libsvtav1": 63}[point.source_codec]
    ]
    assert harsh
    assert all(point.delta_median < 0 for point in harsh)
    mild = [
        point
        for point in points
        if point.source_crf == {"libx264": 18, "libx265": 18, "libsvtav1": 28}[point.source_codec]
    ]
    assert any(point.delta_median > 0 for point in mild)
