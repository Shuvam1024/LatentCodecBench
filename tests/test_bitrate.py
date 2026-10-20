"""Latent bitrate: uniform quantization, zstd size, and the pixel denominator.

These tests do not load a checkpoint and do not need a GPU. The sweep
test injects a measurer, the same way the reconstruction tests do.
"""

from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentcodecbench.analysis.table import read_results
from latentcodecbench.codecs.rate import bits_per_pixel
from latentcodecbench.vaes import GEOMETRIES
from latentcodecbench.vaes.bitrate import (
    FULL_SWEEP_ROWS,
    BitrateError,
    DepthScore,
    load_bitrate_config,
    merge_bitrate,
    run_bitrate,
)
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER
from latentcodecbench.vaes.quantize import (
    BIT_DEPTHS,
    gaussian_cross_entropy_bits,
    measure_latent,
    parse_payload,
    pixel_rate,
)
from latentcodecbench.vaes.scale import denormalize, normalize, published_scale
from latentcodecbench.vaes.smoke import tiling_flags
from latentcodecbench.vaes.wrapper import VideoVae

_MANIFEST = """\
version: 1
canonical:
  width: 832
  height: 480
  num_frames: 33
  pix_fmt: yuv420p
clips:
{clips}
"""

_CLIP = """\
  - id: {clip_id}
    source_url: file:///tmp/{clip_id}.y4m
    sha256: {sha}
    license: test
    license_url: https://example.com/license
    native_width: 832
    native_height: 480
    native_fps_num: 24
    native_fps_den: 1
    category: test
"""


def _require_vae() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        if os.environ.get("LATENTCODECBENCH_REQUIRE_VAE") == "1":
            pytest.fail("CI requires the vae extra; torch is not installed")
        pytest.importorskip("torch")
    raise AssertionError("unreachable")


def _repo_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vae-bitrate.yaml"


def _write_manifest(directory: Path, clip_ids: list[str]) -> Path:
    clips = "\n".join(_CLIP.format(clip_id=clip_id, sha="ab" * 32) for clip_id in clip_ids)
    path = directory / "clips.yaml"
    path.write_text(_MANIFEST.format(clips=clips), encoding="utf-8")
    clip_dir = directory / "clips"
    clip_dir.mkdir()
    for clip_id in clip_ids:
        (clip_dir / f"{clip_id}.y4m").write_bytes(b"")
    return path


class _Module:
    def __init__(self) -> None:
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

    def eval(self) -> _Module:
        return self

    def enable_tiling(self) -> None:
        self.use_tiling = True


def _loader(name: str) -> VideoVae:
    module = _Module()
    if name == "hunyuan":
        module.use_framewise_encoding = True
        module.use_framewise_decoding = True
    return VideoVae(GEOMETRIES[name], module)


def test_published_scales_match_the_pinned_channel_counts() -> None:
    wan = published_scale("wan2_1")
    assert len(wan.shift) == 16
    assert wan.shift[0] == pytest.approx(-0.7571)
    assert published_scale("wan2_2").shift[0] == pytest.approx(-0.2289)
    assert len(published_scale("wan2_2").scale) == 48
    hunyuan = published_scale("hunyuan")
    assert hunyuan.shift == (0.0,) * 16
    assert hunyuan.scale[0] == pytest.approx(0.476986)
    ltx = published_scale("ltx")
    assert len(ltx.shift) == 128
    assert ltx.shift[0] == pytest.approx(0.0062260678969323635)


def test_normalize_roundtrips_and_hunyuan_is_a_scalar_gain() -> None:
    latent = np.linspace(-1.0, 1.0, 16 * 2 * 2 * 2, dtype=np.float64).reshape(16, 2, 2, 2)
    scale = published_scale("hunyuan")
    normalized = normalize(latent, scale)
    assert normalized[0, 0, 0, 0] == pytest.approx(latent[0, 0, 0, 0] * 0.476986)
    assert np.allclose(denormalize(normalized, scale), latent)

    wan = np.zeros((16, 1, 1, 1), dtype=np.float64)
    wan_scale = published_scale("wan2_1")
    expected = -wan_scale.shift[0] * wan_scale.scale[0]
    assert normalize(wan, wan_scale)[0, 0, 0, 0] == pytest.approx(expected)
    assert np.allclose(denormalize(normalize(wan, wan_scale), wan_scale), wan)


def test_quantizer_hits_endpoints_and_zstd_counts_the_header() -> None:
    scale = published_scale("wan2_1")
    # The published mean normalizes to 0, the middle of [-4, 4].
    latent = np.broadcast_to(scale.shift_array(), (16, 2, 2, 2)).copy()
    payload, restored = measure_latent(latent, scale, 4, low=-4.0, high=4.0)
    assert payload.bits == 8 * len(payload.payload)
    assert payload.bits - 8 * len(payload.compressed) == 8 * 24
    parsed, codes = parse_payload(payload.payload)
    assert np.array_equal(codes, payload.codes)
    assert parsed.bits == payload.bits
    assert int(payload.codes.flat[0]) == 8
    on_grid = dequantized_roundtrip(payload.codes, 4)
    assert np.allclose(on_grid, dequantized_roundtrip(quantize_nearest(on_grid, 4), 4))
    assert restored.shape == latent.shape


def test_bpp_uses_pixel_count_not_the_latent_grid() -> None:
    scale = published_scale("ltx")
    latent = np.random.default_rng(0).normal(size=(128, 2, 2, 2))
    payload, _restored = measure_latent(latent, scale, 8)
    width, height, frames = 32, 16, 9
    bpp, kbps = pixel_rate(
        payload.bits, width=width, height=height, frames=frames, fps_num=30, fps_den=1
    )
    pixels = width * height * frames
    latent_cells = int(np.prod(latent.shape))
    assert bpp == bits_per_pixel(payload.bits, width, height, frames)
    assert bpp == pytest.approx(payload.bits / pixels)
    assert bpp != pytest.approx(payload.bits / latent_cells)
    assert kbps == pytest.approx((payload.bits / (frames / 30)) / 1000.0)


def test_dropped_channels_arrive_at_the_decoder_as_zeros() -> None:
    scale = published_scale("hunyuan")
    latent = np.ones((16, 1, 2, 2), dtype=np.float64)
    _payload, restored = measure_latent(latent, scale, 8, channels=1)
    assert np.count_nonzero(restored[1:]) == 0
    assert np.any(restored[0] != 0)


def test_gaussian_estimate_is_not_a_file_size() -> None:
    scale = published_scale("wan2_1")
    latent = np.zeros((16, 2, 2, 2), dtype=np.float64)
    payload, _restored = measure_latent(latent, scale, 2)
    center = int(np.rint(np.median(payload.codes)))
    tail = np.zeros_like(payload.codes)
    middle = np.full_like(payload.codes, center)
    tail_bits = gaussian_cross_entropy_bits(tail, 2, low=-4.0, high=4.0)
    center_bits = gaussian_cross_entropy_bits(middle, 2, low=-4.0, high=4.0)
    assert center_bits < tail_bits
    assert isinstance(payload.entropy_bits, float)
    assert payload.entropy_bits != payload.bits
    from latentcodecbench.vaes.quantize import dequantize_codes

    levels = 4
    codes = np.arange(levels, dtype=np.uint16)
    edges_ok = gaussian_cross_entropy_bits(codes, 2, low=-4.0, high=4.0)
    assert edges_ok > 0
    endpoints = dequantize_codes(np.array([0, levels - 1]), 2, low=-4.0, high=4.0)
    assert endpoints.tolist() == [-4.0, 4.0]


def test_deeper_codes_of_noise_are_a_larger_file() -> None:
    scale = published_scale("wan2_2")
    latent = np.random.default_rng(1).normal(size=(48, 2, 4, 4))
    small, _ = measure_latent(latent, scale, 2)
    large, _ = measure_latent(latent, scale, 12)
    assert large.bits > small.bits


def test_packaged_config_matches_the_repo_file() -> None:
    import latentcodecbench.vaes as vaes

    packaged = Path(vaes.__file__).resolve().parent / "bitrate_config.yaml"
    assert packaged.read_bytes() == _repo_config().read_bytes()
    config = load_bitrate_config(packaged)
    assert config.vaes == SMOKE_ORDER
    assert config.bit_depths == BIT_DEPTHS
    assert config.channels is None
    assert FULL_SWEEP_ROWS == 34 * 4 * 6


def test_sweep_writes_rate_columns_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_vae()
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "b" * 40)
    manifest = _write_manifest(tmp_path, ["alpha"])
    out = tmp_path / "vae-bitrate.csv"

    def measurer(vae: VideoVae, _source: Path, _work: Path, config: Any) -> list[DepthScore]:
        flags = tiling_flags(vae.module)
        rows = []
        for depth in config.bit_depths:
            bits = 1000 * depth
            rows.append(
                DepthScore(
                    bit_depth=depth,
                    bits=bits,
                    bpp=bits_per_pixel(bits, 832, 480, 33),
                    kbps=1.5,
                    entropy_bits=float(bits) / 2,
                    encode_seconds=0.2,
                    decode_seconds=0.3,
                    peak_vram_bytes=1024,
                    spatial_tiling=flags["spatial_tiling"],
                    framewise_encoding=flags["framewise_encoding"],
                    framewise_decoding=flags["framewise_decoding"],
                    retried_with_tiling=False,
                    chunk_count=3,
                    per_chunk_seconds=0.01,
                    latent_shape="16x2x2x2",
                    psnr_y=30.0 + depth,
                    psnr_rgb=29.0,
                    ssim=0.9,
                    vmaf=80.0,
                    flicker=0.1,
                    width=832,
                    height=480,
                    frames=33,
                    fps_num=24,
                    fps_den=1,
                )
            )
        return rows

    moment = datetime(2026, 10, 20, tzinfo=UTC)
    written = run_bitrate(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.11.0",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        now=moment,
        measurer=measurer,
        row_timeout=0,
        heartbeat=0,
    )
    assert len(written) == 4 * len(BIT_DEPTHS)
    assert {row.crf for row in written} == set(BIT_DEPTHS)
    assert {row.preset for row in written} == {"zstd"}
    assert all(row.lpips is None for row in written)
    assert all(row.bits is not None and row.entropy_bits is not None for row in written)
    sample = next(row for row in written if row.codec == "wan2_1" and row.crf == 8)
    assert sample.bits is not None and sample.bpp is not None
    assert sample.bpp == pytest.approx(sample.bits / (832 * 480 * 33))
    again = run_bitrate(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.11.0",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        measurer=measurer,
        row_timeout=0,
        heartbeat=0,
    )
    assert len(again) == len(written)
    assert read_results(out)[0].run_id == written[0].run_id


def test_merge_refuses_a_conflicting_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_vae()
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "c" * 40)
    manifest = _write_manifest(tmp_path, ["alpha"])
    out = tmp_path / "vae-bitrate.csv"

    def measurer(vae: VideoVae, _source: Path, _work: Path, config: Any) -> list[DepthScore]:
        flags = tiling_flags(vae.module)
        return [
            DepthScore(
                bit_depth=depth,
                bits=8000,
                bpp=bits_per_pixel(8000, 832, 480, 33),
                kbps=2.0,
                entropy_bits=100.0,
                encode_seconds=0.2,
                decode_seconds=0.3,
                peak_vram_bytes=1024,
                spatial_tiling=flags["spatial_tiling"],
                framewise_encoding=flags["framewise_encoding"],
                framewise_decoding=flags["framewise_decoding"],
                retried_with_tiling=False,
                chunk_count=2,
                per_chunk_seconds=0.01,
                latent_shape="16x1x1x1",
                psnr_y=28.0,
                psnr_rgb=27.0,
                ssim=0.8,
                vmaf=70.0,
                flicker=0.2,
                width=832,
                height=480,
                frames=33,
                fps_num=24,
                fps_den=1,
            )
            for depth in config.bit_depths
        ]

    run_bitrate(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.11.0",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        measurer=measurer,
        row_timeout=0,
        heartbeat=0,
    )
    dest = tmp_path / "docs" / "vae-bitrate.csv"
    merged = merge_bitrate(out, dest, config_path=_repo_config())
    assert len(merged) == 24
    changed = tmp_path / "changed.csv"
    changed.write_text(out.read_text(encoding="utf-8").replace("28.0", "29.0", 1), encoding="utf-8")
    with pytest.raises(BitrateError, match="different row"):
        merge_bitrate(changed, dest, config_path=_repo_config())


def test_notebook_states_the_colab_contract() -> None:
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "vae_bitrate.ipynb"
    text = notebook.read_text(encoding="utf-8")
    for needle in (
        "torchao",
        "--force-reinstall",
        "latentcodecbench._build_info",
        "bit_depths",
        "zstd",
        "entropy_bits",
        "/content/results/vae-bitrate.csv",
        "/content/drive/MyDrive/LatentCodecBench",
        "scripts/merge-vae-bitrate.py",
        "configs/vae-bitrate.yaml",
    ):
        assert needle in text


def quantize_nearest(values: np.ndarray, bit_depth: int) -> np.ndarray:
    from latentcodecbench.vaes.quantize import quantize_codes

    return quantize_codes(values, bit_depth, low=-4.0, high=4.0)


def dequantized_roundtrip(codes: np.ndarray, bit_depth: int) -> np.ndarray:
    from latentcodecbench.vaes.quantize import dequantize_codes

    return dequantize_codes(codes, bit_depth, low=-4.0, high=4.0)
