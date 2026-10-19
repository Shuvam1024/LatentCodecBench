"""Known-value tests for PSNR, SSIM, flicker, VMAF, and the LPIPS skip."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from latentcodecbench.data.preprocess import ffmpeg_binary
from latentcodecbench.data.y4m import write_y4m
from latentcodecbench.metrics.color import rgb_to_ycbcr, ycbcr_to_rgb
from latentcodecbench.metrics.flicker import temporal_flicker
from latentcodecbench.metrics.perceptual import MetricUnavailable, lpips_distance, maybe_lpips
from latentcodecbench.metrics.psnr import psnr
from latentcodecbench.metrics.ssim import ssim
from latentcodecbench.metrics.vmaf import vmaf


def test_psnr_is_infinite_when_inputs_match() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert psnr(image, image) == math.inf


def test_psnr_of_a_single_level_step() -> None:
    # 2x2, one sample off by 1. MSE = 0.25. PSNR = 10 log10(255^2 / 0.25).
    reference = np.zeros((2, 2), dtype=np.uint8)
    distorted = reference.copy()
    distorted[0, 0] = 1
    assert psnr(reference, distorted) == pytest.approx(54.15140352195873)


def test_psnr_of_a_constant_offset() -> None:
    # Every sample is 10 away. MSE = 100.
    reference = np.zeros((4, 4), dtype=np.uint8)
    distorted = np.full((4, 4), 10, dtype=np.uint8)
    assert psnr(reference, distorted) == pytest.approx(10.0 * math.log10(65025.0 / 100.0))


def test_ssim_of_identical_planes_is_one() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
    assert ssim(image, image) == pytest.approx(1.0)


def test_ssim_of_two_flat_fields_matches_the_closed_form() -> None:
    # Variances are 0, so SSIM reduces to (2 mu_x mu_y + C1) / (mu_x^2 + mu_y^2 + C1).
    # C1 = (0.01 * 255)^2.
    reference = np.zeros((16, 16), dtype=np.uint8)
    distorted = np.full((16, 16), 255, dtype=np.uint8)
    c1 = (0.01 * 255.0) ** 2
    expected = c1 / (255.0**2 + c1)
    assert ssim(reference, distorted) == pytest.approx(expected)


def test_ssim_matches_an_independently_computed_value() -> None:
    # Locked against scikit-image structural_similarity with gaussian_weights,
    # sigma 1.5, win_size 11, use_sample_covariance False, data_range 255.
    # The left half of the frame is an unrelated field. scikit-image is not
    # a dependency; this number is the check.
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 256, size=(20, 18), dtype=np.uint8)
    distorted = reference.copy()
    other = np.random.default_rng(2).integers(0, 256, size=(20, 18), dtype=np.uint8)
    distorted[:, :9] = other[:, :9]
    assert ssim(reference, distorted) == pytest.approx(0.4975158429069607)


def test_flicker_is_zero_for_identical_clips() -> None:
    clip = np.arange(24, dtype=np.uint8).reshape(3, 2, 4)
    assert temporal_flicker(clip, clip) == 0.0


def test_flicker_ignores_a_constant_offset() -> None:
    reference = np.zeros((4, 2, 2), dtype=np.uint8)
    reference[2:] = 40
    distorted = reference + np.uint8(7)
    assert temporal_flicker(reference, distorted) == 0.0


def test_flicker_of_a_known_jump() -> None:
    # Two frames, one sample. Reference stays at 0. Distorted jumps by 10.
    reference = np.zeros((2, 1, 1), dtype=np.uint8)
    distorted = np.zeros((2, 1, 1), dtype=np.uint8)
    distorted[1, 0, 0] = 10
    assert temporal_flicker(reference, distorted) == pytest.approx(10.0)


def test_limited_range_black_and_white() -> None:
    black_y = np.full((1, 2, 2), 16, dtype=np.uint8)
    white_y = np.full((1, 2, 2), 235, dtype=np.uint8)
    neutral = np.full((1, 1, 1), 128, dtype=np.uint8)
    black = ycbcr_to_rgb(black_y, neutral, neutral)
    white = ycbcr_to_rgb(white_y, neutral, neutral)
    assert np.allclose(black, 0.0)
    assert np.allclose(white, 255.0)


def test_gray_luma_roundtrips_exactly() -> None:
    luma = np.full((1, 2, 2), 126, dtype=np.uint8)
    chroma = np.full((1, 1, 1), 128, dtype=np.uint8)
    back, cb, cr = rgb_to_ycbcr(ycbcr_to_rgb(luma, chroma, chroma))
    assert np.array_equal(back, luma)
    assert np.array_equal(cb, chroma)
    assert np.array_equal(cr, chroma)


def test_unclipped_block_roundtrips_luma_and_chroma() -> None:
    # One 2x2 block. Luma varies, chroma is shared, and the RGB stays inside 0..255.
    luma = np.array([[[80, 90], [100, 110]]], dtype=np.uint8)
    cb = np.array([[[120]]], dtype=np.uint8)
    cr = np.array([[[140]]], dtype=np.uint8)
    rgb = ycbcr_to_rgb(luma, cb, cr)
    assert np.all((rgb > 0.0) & (rgb < 255.0))
    back, back_cb, back_cr = rgb_to_ycbcr(rgb)
    assert np.array_equal(back, luma)
    assert np.array_equal(back_cb, cb)
    assert np.array_equal(back_cr, cr)


def test_lpips_reports_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("latentcodecbench.metrics.perceptual.importlib.import_module", explode)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(MetricUnavailable):
        lpips_distance(rgb, rgb)
    assert maybe_lpips(rgb, rgb) is None


def _ffmpeg_has_libvmaf() -> bool:
    try:
        binary = ffmpeg_binary()
    except RuntimeError:
        return False
    completed = subprocess.run(
        [binary, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "libvmaf" in completed.stdout


@pytest.mark.skipif(not _ffmpeg_has_libvmaf(), reason="ffmpeg was built without libvmaf")
def test_vmaf_of_identical_clips_is_near_100(tmp_path: Path) -> None:
    # A flat ramp scores in the mid-97s: libvmaf is a perceptual model, not
    # an identity. testsrc2 is textured enough that a match sits above 99.
    path = tmp_path / "clip.y4m"
    subprocess.run(
        [
            ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25",
            "-frames:v",
            "8",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    assert vmaf(path, path) > 99.0


@pytest.mark.skipif(not _ffmpeg_has_libvmaf(), reason="ffmpeg was built without libvmaf")
def test_vmaf_drops_when_luma_is_shifted(tmp_path: Path) -> None:
    luma = np.tile(np.linspace(32, 220, 160, dtype=np.uint8), (5, 90, 1))
    chroma = np.full((5, 45, 80), 128, dtype=np.uint8)
    reference = tmp_path / "ref.y4m"
    distorted = tmp_path / "dist.y4m"
    write_y4m(reference, luma, chroma, chroma, fps_num=25, fps_den=1)
    shifted = np.clip(luma.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    write_y4m(distorted, shifted, chroma, chroma, fps_num=25, fps_den=1)
    assert vmaf(distorted, reference) < vmaf(reference, reference)


def test_vmaf_error_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.y4m"
    with pytest.raises(FileNotFoundError):
        vmaf(missing, missing)
