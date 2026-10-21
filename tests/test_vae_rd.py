"""VAE BD-rate reuses the PCHIP integral and matches the committed table."""

from __future__ import annotations

from pathlib import Path

import pytest

from latentcodecbench.analysis.bdrate import bd_rate
from latentcodecbench.analysis.table import ResultRow, read_results
from latentcodecbench.analysis.vae_rd import (
    EXCLUDED_BIT_DEPTH,
    plot_vae_rd,
    render_bdrate_csv,
    vae_bd_rate_markdown,
    vae_clip_bd_rates,
)

_ROOT = Path(__file__).resolve().parents[1]


def _row(
    clip_id: str,
    codec: str,
    crf: int,
    bpp: float,
    psnr: float,
    *,
    vmaf: float = 80.0,
) -> ResultRow:
    return ResultRow(
        run_id="run",
        git_sha="a" * 40,
        config_sha256="b" * 64,
        clip_id=clip_id,
        codec=codec,
        preset="medium",
        crf=crf,
        width=832,
        height=480,
        frames=33,
        fps_num=25,
        fps_den=1,
        bits=int(bpp * 832 * 480 * 33),
        bpp=bpp,
        kbps=1.0,
        encode_seconds=0.1,
        decode_seconds=0.1,
        psnr_y=psnr,
        psnr_rgb=psnr,
        ssim=0.9,
        vmaf=vmaf,
        flicker=0.1,
        lpips=None,
    )


def test_half_rate_curve_is_minus_50_and_depth_2_is_ignored() -> None:
    anchor_rate = [1000.0, 500.0, 200.0, 80.0]
    anchor_psnr = [42.0, 38.0, 34.0, 30.0]
    rows = [
        _row("clip", "libx264", crf, rate, psnr)
        for crf, rate, psnr in zip((18, 22, 26, 30), anchor_rate, anchor_psnr, strict=True)
    ]
    rows.extend(
        _row("clip", "wan2_1", crf, rate / 2, psnr)
        for crf, rate, psnr in zip((4, 6, 8, 10), anchor_rate, anchor_psnr, strict=True)
    )
    # A depth-2 sample below the classic floor would bend the cubic if it
    # were kept. The reported rate stays the four-point half-rate case.
    rows.append(_row("clip", "wan2_1", EXCLUDED_BIT_DEPTH, 10.0, 12.0))
    found = [
        row
        for row in vae_clip_bd_rates(rows)
        if row.codec == "wan2_1" and row.anchor == "libx264" and row.metric == "psnr_y"
    ]
    assert len(found) == 1
    assert found[0].status == "defined"
    half = [rate / 2 for rate in anchor_rate]
    assert found[0].bd_rate == bd_rate(anchor_rate, anchor_psnr, half, anchor_psnr)
    assert found[0].bd_rate == pytest.approx(-50.0)
    kept = bd_rate(anchor_rate, anchor_psnr, [10.0, *half], [12.0, *anchor_psnr])
    assert kept != pytest.approx(-50.0)


def test_no_overlap_and_a_reversal_stay_undefined() -> None:
    rows = [
        _row("clip", "libx264", crf, rate, psnr)
        for crf, rate, psnr in zip(
            (18, 22, 26, 30),
            (0.4, 0.2, 0.1, 0.05),
            (42.0, 38.0, 34.0, 30.0),
            strict=True,
        )
    ]
    rows.extend(
        _row("clip", "ltx", crf, rate, psnr)
        for crf, rate, psnr in zip(
            (4, 6, 8, 10),
            (0.2, 0.15, 0.1, 0.08),
            (20.0, 18.0, 16.0, 14.0),
            strict=True,
        )
    )
    rows.extend(
        _row("other", "libx264", crf, rate, psnr)
        for crf, rate, psnr in zip(
            (18, 22, 26, 30),
            (0.4, 0.2, 0.1, 0.05),
            (42.0, 38.0, 34.0, 30.0),
            strict=True,
        )
    )
    # Quality falls as rate rises, so the integral is refused.
    rows.extend(
        _row("other", "ltx", crf, rate, psnr)
        for crf, rate, psnr in zip(
            (4, 6, 8, 10),
            (0.05, 0.1, 0.2, 0.4),
            (30.0, 34.0, 38.0, 36.0),
            strict=True,
        )
    )
    by_clip = {
        (row.clip_id, row.metric): row
        for row in vae_clip_bd_rates(rows)
        if row.codec == "ltx" and row.anchor == "libx264" and row.metric == "psnr_y"
    }
    assert by_clip[("clip", "psnr_y")].status == "no_overlap"
    assert by_clip[("clip", "psnr_y")].bd_rate is None
    assert by_clip[("other", "psnr_y")].status == "nonmonotonic"
    assert by_clip[("other", "psnr_y")].bd_rate is None


def test_committed_table_matches_the_csvs_and_the_note() -> None:
    rows = read_results(_ROOT / "docs" / "results" / "classic-baselines.csv")
    rows.extend(read_results(_ROOT / "docs" / "results" / "vae-bitrate.csv"))
    clips = vae_clip_bd_rates(rows)
    csv_path = _ROOT / "docs" / "results" / "vae-rd-bdrate.csv"
    assert csv_path.read_text(encoding="utf-8") == render_bdrate_csv(clips)
    note = (_ROOT / "docs" / "learn" / "09-vae-rd-curves.md").read_text(encoding="utf-8")
    assert vae_bd_rate_markdown(rows) in note
    assert sum(row.status == "defined" for row in clips) > 0
    assert all(row.bd_rate is None for row in clips if row.status != "defined")


def test_committed_figures_exist() -> None:
    for name in ("rd-psnr.png", "rd-vmaf.png", "rd-bdrate-clips.png"):
        path = _ROOT / "docs" / "results" / name
        assert path.is_file()
        assert path.stat().st_size > 1000


def test_plots_write_three_figures(tmp_path: Path) -> None:
    rows = read_results(_ROOT / "docs" / "results" / "classic-baselines.csv")
    rows.extend(read_results(_ROOT / "docs" / "results" / "vae-bitrate.csv"))
    paths = plot_vae_rd(rows, tmp_path)
    assert [path.name for path in paths] == ["rd-psnr.png", "rd-vmaf.png", "rd-bdrate-clips.png"]
    assert all(path.stat().st_size > 1000 for path in paths)
