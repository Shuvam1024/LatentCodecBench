"""Results table, config lock, BD-rate aggregation, and one real encode."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from latentcodecbench.analysis.config import load_config
from latentcodecbench.analysis.plot import clip_overlaps, mean_bd_rate, plot_rd, write_overlap
from latentcodecbench.analysis.run import run_baselines
from latentcodecbench.analysis.table import ResultRow, read_results, write_results
from latentcodecbench.codecs.baseline import (
    BENCHMARK_PRESET,
    SVT_AV1_CRF,
    X264_CRF,
    X265_CRF,
)
from latentcodecbench.data.preprocess import ffmpeg_binary
from latentcodecbench.data.y4m import write_y4m


def _row(
    clip_id: str,
    codec: str,
    crf: int,
    bpp: float,
    psnr: float,
    vmaf: float,
) -> ResultRow:
    return ResultRow(
        run_id="20261007T000000Z-abc123def456-0123456789ab",
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
        kbps=100.0,
        encode_seconds=0.1,
        decode_seconds=0.05,
        psnr_y=psnr,
        psnr_rgb=psnr - 1,
        ssim=0.9,
        vmaf=vmaf,
        flicker=0.2,
        lpips=None,
    )


def test_classic_config_matches_the_codec_constants() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "classic.yaml")
    by_name = {codec.name: codec for codec in config.codecs}
    assert by_name["libx264"].crf == X264_CRF
    assert by_name["libx265"].crf == X265_CRF
    assert by_name["libsvtav1"].crf == SVT_AV1_CRF
    assert by_name["libx264"].preset == BENCHMARK_PRESET["libx264"]
    assert by_name["libx265"].preset == BENCHMARK_PRESET["libx265"]
    assert by_name["libsvtav1"].preset == BENCHMARK_PRESET["libsvtav1"]


def test_table_sorts_and_roundtrips_an_empty_lpips(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    rows = [
        _row("beta", "libx265", 28, 0.05, 35.0, 80.0),
        _row("alpha", "libx264", 18, 0.20, 42.0, 95.0),
        _row("alpha", "libx264", 28, 0.08, 36.0, 80.0),
    ]
    write_results(path, rows)
    loaded = read_results(path)
    assert [(row.clip_id, row.codec, row.crf) for row in loaded] == [
        ("alpha", "libx264", 18),
        ("alpha", "libx264", 28),
        ("beta", "libx265", 28),
    ]
    assert loaded[0].lpips is None
    assert loaded[0].bpp == pytest.approx(0.20)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "peak_vram_bytes" not in header.split(",")


def test_mean_bd_rate_of_a_half_rate_curve() -> None:
    psnr = [42.0, 38.0, 34.0, 30.0]
    vmaf = [95.0, 85.0, 70.0, 50.0]
    crf = [18, 23, 28, 33]
    anchor_bpp = [0.40, 0.20, 0.10, 0.05]
    rows: list[ResultRow] = []
    for point, bpp, score, quality in zip(crf, anchor_bpp, psnr, vmaf, strict=True):
        rows.append(_row("clip", "libx264", point, bpp, score, quality))
        rows.append(_row("clip", "libx265", point, bpp / 2, score, quality))
    summaries = mean_bd_rate(rows, anchor="libx264", metric="psnr_y")
    assert summaries[0].codec == "libx265"
    assert summaries[0].mean_bd_rate == pytest.approx(-50.0)
    assert summaries[0].clips == 1
    assert summaries[0].skipped == 0
    assert summaries[0].mean_overlap == pytest.approx(1.0)
    assert summaries[0].below_overlap == 0


def test_overlap_table_flags_a_short_shared_range(tmp_path: Path) -> None:
    # Anchor covers 28..40 dB. Test covers 36..44. Intersection 4, union 16.
    rows = [
        _row("clip", "libx264", crf, bpp, psnr, vmaf)
        for crf, bpp, psnr, vmaf in (
            (18, 0.40, 40.0, 90.0),
            (22, 0.20, 36.0, 80.0),
            (26, 0.10, 32.0, 60.0),
            (30, 0.05, 28.0, 40.0),
        )
    ]
    rows.extend(
        _row("clip", "libsvtav1", crf, bpp, psnr, vmaf)
        for crf, bpp, psnr, vmaf in (
            (28, 0.30, 44.0, 92.0),
            (35, 0.20, 42.0, 91.0),
            (42, 0.10, 40.0, 90.0),
            (49, 0.05, 36.0, 88.0),
        )
    )
    found = clip_overlaps(rows)
    psnr = next(row for row in found if row.metric == "psnr_y")
    assert psnr.overlap == pytest.approx(4.0 / 16.0)
    path = tmp_path / "overlap.csv"
    write_overlap(path, found)
    text = path.read_text(encoding="utf-8")
    assert "below_0_75" in text.splitlines()[0]
    assert ",1\n" in text or text.endswith(",1\n")


def test_plot_writes_both_figures(tmp_path: Path) -> None:
    rows = [
        _row("clip", "libx264", crf, bpp, psnr, vmaf)
        for crf, bpp, psnr, vmaf in (
            (18, 0.4, 42.0, 95.0),
            (23, 0.2, 38.0, 85.0),
            (28, 0.1, 34.0, 70.0),
            (33, 0.05, 30.0, 50.0),
        )
    ]
    psnr_path, vmaf_path = plot_rd(rows, tmp_path)
    assert psnr_path.is_file() and psnr_path.stat().st_size > 1000
    assert vmaf_path.is_file() and vmaf_path.stat().st_size > 1000


def _has_tools() -> bool:
    try:
        binary = ffmpeg_binary()
    except RuntimeError:
        return False
    completed = subprocess.run(
        [binary, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    filters = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "libx264" in completed.stdout and "libvmaf" in filters.stdout


@pytest.mark.skipif(not _has_tools(), reason="ffmpeg needs libx264 and libvmaf")
def test_run_encodes_one_point_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "c" * 40)
    luma = np.random.default_rng(0).integers(0, 256, size=(8, 96, 160), dtype=np.uint8)
    chroma = np.full((8, 48, 80), 128, dtype=np.uint8)
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    write_y4m(clip_dir / "tiny.y4m", luma, chroma, chroma, fps_num=25, fps_den=1)
    manifest = tmp_path / "clips.yaml"
    manifest.write_text(
        """
version: 1
canonical:
  width: 832
  height: 480
  num_frames: 33
  pix_fmt: yuv420p
clips:
  - id: tiny
    source_url: file:///tmp/tiny.y4m
    sha256: {sha}
    license: test
    license_url: https://example.com/license
    native_width: 160
    native_height: 96
    native_fps_num: 25
    native_fps_den: 1
    category: test
""".format(sha="ab" * 32),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "version: 1\ncodecs:\n  libx264:\n    preset: ultrafast\n    crf: [18, 28, 38, 43]\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    moment = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    rows = run_baselines(
        manifest_path=manifest,
        config_path=config,
        clip_dir=clip_dir,
        out=out,
        work_dir=tmp_path / "work",
        now=moment,
    )
    assert len(rows) == 4
    assert rows[0].run_id.startswith("20261007T120000Z-cccccccccccc-")
    assert rows[0].psnr_y > 0
    assert rows[0].vmaf > 0
    again = run_baselines(
        manifest_path=manifest,
        config_path=config,
        clip_dir=clip_dir,
        out=out,
        work_dir=tmp_path / "work",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert [row.crf for row in again] == [row.crf for row in rows]
    assert again[0].run_id == rows[0].run_id
