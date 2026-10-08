"""Codec wrappers: rate math, thread flags, and a tiny CRF encode."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from latentcodecbench.codecs.baseline import (
    BENCHMARK_PRESET,
    SVT_AV1_CRF,
    X264_CRF,
    X265_CRF,
    encode_y4m,
    encoder_args,
)
from latentcodecbench.codecs.rate import bits_per_pixel, kilobits_per_second
from latentcodecbench.data.preprocess import ffmpeg_binary
from latentcodecbench.data.y4m import write_y4m


def test_bpp_divides_by_displayed_samples() -> None:
    bits = 100_000
    assert bits_per_pixel(bits, 832, 480, 33) == pytest.approx(bits / (832 * 480 * 33))


def test_kbps_uses_the_source_frame_rate() -> None:
    # 33 frames at 25 fps is 1.32 seconds.
    bits = 132_000
    assert kilobits_per_second(bits, 33, 25, 1) == pytest.approx(100.0)


def test_crf_ladders_have_six_increasing_points() -> None:
    assert X264_CRF == (18, 22, 26, 30, 34, 38)
    assert X265_CRF == (18, 22, 26, 30, 34, 38)
    assert SVT_AV1_CRF == (28, 35, 42, 49, 56, 63)
    assert BENCHMARK_PRESET == {"libx264": "medium", "libx265": "medium", "libsvtav1": "6"}


def test_encoders_are_single_threaded_and_not_psnr_tuned() -> None:
    x264 = encoder_args("libx264", 28, "medium")
    assert x264[x264.index("-threads") + 1] == "1"
    assert "sliced-threads=0" in x264
    assert "-tune" not in x264

    x265 = encoder_args("libx265", 28, "medium")
    assert "pools=1:frame-threads=1" in x265
    assert "-tune" not in x265

    svt = encoder_args("libsvtav1", 35, "6")
    assert "lp=1:tune=0" in svt


def _has_encoder(name: str) -> bool:
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
    return name in completed.stdout


def _tiny_source(path: Path) -> None:
    # Noise, not a flat field: ultrafast on a constant collapses every CRF
    # to the same handful of header bits.
    rng = np.random.default_rng(0)
    luma = rng.integers(0, 256, size=(6, 64, 64), dtype=np.uint8)
    chroma = rng.integers(0, 256, size=(6, 32, 32), dtype=np.uint8)
    write_y4m(path, luma, chroma, chroma, fps_num=25, fps_den=1)


@pytest.mark.skipif(not _has_encoder("libx264"), reason="ffmpeg has no libx264")
def test_lower_crf_spends_more_bits(tmp_path: Path) -> None:
    source = tmp_path / "src.y4m"
    _tiny_source(source)
    fine = encode_y4m(source, tmp_path / "fine.mp4", codec="libx264", crf=18, preset="ultrafast")
    coarse = encode_y4m(
        source, tmp_path / "coarse.mp4", codec="libx264", crf=40, preset="ultrafast"
    )
    assert fine.bits > coarse.bits
    assert fine.bpp == pytest.approx(fine.bits / (64 * 64 * 6))
    assert fine.kbps == pytest.approx(fine.bits / (6 / 25) / 1000)
    assert fine.encode_seconds > 0
    assert fine.decode_seconds > 0
    assert fine.frames == 6


@pytest.mark.skipif(
    not all(_has_encoder(name) for name in ("libx265", "libsvtav1")),
    reason="ffmpeg is missing libx265 or libsvtav1",
)
def test_hevc_and_av1_encode_a_tiny_clip(tmp_path: Path) -> None:
    source = tmp_path / "src.y4m"
    _tiny_source(source)
    hevc = encode_y4m(source, tmp_path / "hevc.mp4", codec="libx265", crf=32, preset="ultrafast")
    av1 = encode_y4m(source, tmp_path / "av1.mp4", codec="libsvtav1", crf=40, preset="12")
    assert hevc.bits > 0
    assert av1.bits > 0
    assert hevc.preset == "ultrafast"
    assert av1.preset == "12"


def test_missing_ffmpeg_encoder_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        encoder_args("libvpx-vp9", 20, "good")
