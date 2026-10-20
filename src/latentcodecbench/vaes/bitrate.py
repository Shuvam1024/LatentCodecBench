"""Bitrate sweep: quantize each posterior mean and score the decode.

One encode is shared by the six bit depths of a clip. ``crf`` stores the
bit depth, because the results table has one operating-point column.
``preset`` is ``zstd``. ``bits`` is the header plus the compressed integer
tensor. ``entropy_bits`` is the factorized Gaussian estimate and is not a
file size. ``lpips`` stays empty. See ADR-0010.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from latentcodecbench.analysis.run import config_sha256, git_sha
from latentcodecbench.analysis.table import (
    ResultRow,
    read_results,
    sort_rows,
    transfer_file,
    write_results,
)
from latentcodecbench.data.manifest import load_manifest
from latentcodecbench.vaes.checkpoints import CHECKPOINTS, SMOKE_ORDER
from latentcodecbench.vaes.quantize import (
    BIT_DEPTHS,
    QUANT_HIGH,
    QUANT_LOW,
    ZSTD_LEVEL,
    measure_latent,
    pixel_rate,
)
from latentcodecbench.vaes.recon import (
    RowTimeout,
    _EncoderTimer,
    _is_drive,
    _is_oom,
    _line_buffered,
    _peak_bytes,
    _prefer_table,
    _read_table,
    _release_cuda,
    _RowWatch,
    _score_pixels,
    _sync,
    _unit_rgb_to_tensor,
    apply_sweep_tiling,
    y4m_to_unit_rgb,
)
from latentcodecbench.vaes.scale import published_scale
from latentcodecbench.vaes.wrapper import VideoVae, _torch

BITRATE_PRESET = "zstd"
FULL_SWEEP_ROWS = 34 * 4 * len(BIT_DEPTHS)

Measurer = Callable[[VideoVae, Path, Path, "BitrateConfig"], list["DepthScore"]]


class BitrateError(RuntimeError):
    """The bitrate sweep could not continue."""


@dataclass(frozen=True)
class BitrateConfig:
    """The hashed sweep settings."""

    vaes: tuple[str, ...]
    bit_depths: tuple[int, ...]
    quant_low: float
    quant_high: float
    zstd_level: int
    channels: int | None


@dataclass(frozen=True)
class DepthScore:
    """One bit depth of one clip, ready for a table row."""

    bit_depth: int
    bits: int
    bpp: float
    kbps: float
    entropy_bits: float
    encode_seconds: float
    decode_seconds: float
    peak_vram_bytes: int | None
    spatial_tiling: bool
    framewise_encoding: bool
    framewise_decoding: bool
    retried_with_tiling: bool
    chunk_count: int | None
    per_chunk_seconds: float | None
    latent_shape: str
    psnr_y: float
    psnr_rgb: float
    ssim: float
    vmaf: float
    flicker: float
    width: int
    height: int
    frames: int
    fps_num: int
    fps_den: int


def load_bitrate_config(path: Path) -> BitrateConfig:
    """Read the sweep config. The VAE list and the bit depths are fixed."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BitrateError(f"{path} must be a mapping")
    if raw.get("version") != 1 or raw.get("kind") != "vae-bitrate":
        raise BitrateError(f"{path} is not a vae-bitrate config")
    if raw.get("hunyuan_spatial_tiling") is not True:
        raise BitrateError(f"{path} must enable Hunyuan spatial tiling")
    names = raw.get("vaes")
    if not isinstance(names, list) or tuple(names) != SMOKE_ORDER:
        joined = ", ".join(SMOKE_ORDER)
        raise BitrateError(f"{path} VAE list must be {joined}")
    depths = raw.get("bit_depths")
    if not isinstance(depths, list) or tuple(int(v) for v in depths) != BIT_DEPTHS:
        raise BitrateError(f"{path} bit depths must be {', '.join(str(v) for v in BIT_DEPTHS)}")
    low = float(raw.get("quant_low", QUANT_LOW))
    high = float(raw.get("quant_high", QUANT_HIGH))
    if not low < high:
        raise BitrateError(f"{path} quantizer range must be increasing")
    level = int(raw.get("zstd_level", ZSTD_LEVEL))
    if level < 1 or level > 22:
        raise BitrateError(f"{path} zstd level must be in 1..22")
    channels = raw.get("channels")
    kept: int | None
    if channels is None:
        kept = None
    else:
        kept = int(channels)
        if kept < 1:
            raise BitrateError(f"{path} channels must be positive")
    metrics = raw.get("metrics")
    if metrics != ["psnr_y", "psnr_rgb", "ssim", "vmaf", "flicker"]:
        raise BitrateError(f"{path} metrics must be the Phase 1 set without LPIPS")
    return BitrateConfig(
        vaes=SMOKE_ORDER,
        bit_depths=BIT_DEPTHS,
        quant_low=low,
        quant_high=high,
        zstd_level=level,
        channels=kept,
    )


def run_bitrate(
    *,
    manifest_path: Path,
    config_path: Path,
    clip_dir: Path,
    out: Path,
    work_dir: Path,
    loader: Callable[[str], VideoVae],
    torch_version: str,
    diffusers_version: str,
    runtime_dtype: str,
    gpu_name: str,
    only: set[str] | None = None,
    now: datetime | None = None,
    measurer: Measurer | None = None,
    row_timeout: float = 900.0,
    heartbeat: float = 60.0,
    publish_timeout: float = 120.0,
) -> list[ResultRow]:
    """Quantize each clip at each bit depth and write ``out`` after every depth.

    ``skip clip vae crf`` is a depth already stored. ``done clip vae crf=N``
    is printed after that depth has been written. A clip that exceeds
    ``row_timeout`` logs ``timeout`` and the sweep continues. When ``out``
    is on Google Drive the table is written on local disk and copied across.
    """
    _line_buffered()
    manifest = load_manifest(manifest_path)
    config = load_bitrate_config(config_path)
    if only is not None:
        known = {clip.id for clip in manifest.clips}
        missing = only - known
        if missing:
            raise BitrateError(f"unknown clip id: {', '.join(sorted(missing))}")
    digest = config_sha256(config_path)
    sha = git_sha()
    work_dir.mkdir(parents=True, exist_ok=True)
    mirror: Path | None = None
    journal = out
    if _is_drive(out):
        mirror = out
        journal = work_dir.parent / out.name
        if _is_drive(journal):
            journal = Path(tempfile.gettempdir()) / out.name
    remote_rows = _read_table(mirror, timeout=publish_timeout) if mirror is not None else []
    local_rows = _read_table(journal, timeout=publish_timeout)
    existing, source = _prefer_table(local_rows, journal, remote_rows, mirror or journal)
    if existing and existing[0].config_sha256 != digest:
        raise BitrateError(f"{source} was built from a different config; refusing to mix rows")
    print(f"resume {len(existing)} rows from {source}", flush=True)
    if existing:
        run_id = existing[0].run_id
    else:
        moment = now if now is not None else datetime.now(UTC)
        run_id = f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{sha[:12]}-{digest[:12]}"
    done = {(row.clip_id, row.codec, row.crf) for row in existing}
    rows = list(existing)
    active = default_measurer if measurer is None else measurer
    for name in config.vaes:
        pending = []
        for clip in manifest.clips:
            if only is not None and clip.id not in only:
                continue
            missing_depths = [
                depth for depth in config.bit_depths if (clip.id, name, depth) not in done
            ]
            if not missing_depths:
                print(f"skip {clip.id} {name}", flush=True)
                continue
            pending.append(clip)
        if not pending:
            continue
        vae: VideoVae | None = None
        try:
            try:
                with _RowWatch(f"load {name}", row_timeout, heartbeat):
                    vae = loader(name)
            except RowTimeout:
                print(f"timeout load {name} after {row_timeout:.3g}s", flush=True)
                continue
            assert vae is not None
            apply_sweep_tiling(name, vae.module)
            revision = CHECKPOINTS[name].revision
            for clip in pending:
                source_y4m = clip_dir / f"{clip.id}.y4m"
                if not source_y4m.is_file():
                    raise BitrateError(f"missing canonical clip {source_y4m}")
                print(f"run {clip.id} {name}", flush=True)
                started = time.monotonic()
                try:
                    with _RowWatch(f"{clip.id} {name}", row_timeout, heartbeat):
                        measured = active(vae, source_y4m, work_dir, config)
                except RowTimeout:
                    print(f"timeout {clip.id} {name} after {row_timeout:.3g}s", flush=True)
                    _release_cuda()
                    continue
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    print(f"fail {clip.id} {name} {detail[:500]}", flush=True)
                    _release_cuda()
                    continue
                for score in measured:
                    if (clip.id, name, score.bit_depth) in done:
                        continue
                    rows.append(
                        _result_row(
                            score,
                            clip_id=clip.id,
                            name=name,
                            revision=revision,
                            run_id=run_id,
                            git_sha=sha,
                            config_sha256=digest,
                            torch_version=torch_version,
                            diffusers_version=diffusers_version,
                            runtime_dtype=runtime_dtype,
                            gpu_name=gpu_name,
                        )
                    )
                    done.add((clip.id, name, score.bit_depth))
                    write_results(journal, rows)
                    published = True
                    if mirror is not None:
                        published = transfer_file(journal, mirror, timeout=publish_timeout) == 0
                    if not published:
                        print(f"publish {mirror} timed out; rows stay in {journal}", flush=True)
                    elapsed = time.monotonic() - started
                    print(
                        f"done {clip.id} {name} crf={score.bit_depth} t={elapsed:.1f}s",
                        flush=True,
                    )
        finally:
            vae = None
            _release_cuda()
    if rows:
        write_results(journal, rows)
        if mirror is not None:
            transfer_file(journal, mirror, timeout=publish_timeout)
    return sort_rows(rows)


def default_measurer(
    vae: VideoVae,
    source: Path,
    work_dir: Path,
    config: BitrateConfig,
) -> list[DepthScore]:
    """Encode once, quantize each bit depth, decode, and score against the y4m."""
    torch = _torch()
    info, rgb = y4m_to_unit_rgb(source)
    frames = int(rgb.shape[0])
    parameter = next(vae.module.parameters())
    video = _unit_rgb_to_tensor(rgb, torch, dtype=parameter.dtype, device=parameter.device)
    del rgb
    encode_seconds, latents, chunk_count, per_chunk, retried, encode_peak = _encode_latent(
        vae, video, torch
    )
    del video
    latent = _latent_numpy(latents)
    shape = "x".join(str(int(v)) for v in latent.shape)
    scores: list[DepthScore] = []
    for depth in config.bit_depths:
        payload, restored = measure_latent(
            latent,
            published_scale(vae.geometry.name),
            depth,
            channels=config.channels,
            low=config.quant_low,
            high=config.quant_high,
            level=config.zstd_level,
        )
        decode_seconds, pixels, decode_peak, decode_retried = _decode_restored(
            vae, restored, torch, parameter
        )
        flags = _flags(vae)
        measured = _score_pixels(
            source,
            info,
            frames,
            pixels,
            work_dir / f"{source.stem}-{vae.geometry.name}-d{depth}.y4m",
        )
        bpp, kbps = pixel_rate(
            payload.bits,
            width=info.width,
            height=info.height,
            frames=frames,
            fps_num=info.fps_num,
            fps_den=info.fps_den,
        )
        scores.append(
            DepthScore(
                bit_depth=depth,
                bits=payload.bits,
                bpp=bpp,
                kbps=kbps,
                entropy_bits=payload.entropy_bits,
                encode_seconds=encode_seconds,
                decode_seconds=decode_seconds,
                peak_vram_bytes=_higher(encode_peak, decode_peak),
                spatial_tiling=flags["spatial_tiling"],
                framewise_encoding=flags["framewise_encoding"],
                framewise_decoding=flags["framewise_decoding"],
                retried_with_tiling=retried or decode_retried,
                chunk_count=chunk_count,
                per_chunk_seconds=per_chunk,
                latent_shape=shape,
                psnr_y=measured.psnr_y,
                psnr_rgb=measured.psnr_rgb,
                ssim=measured.ssim,
                vmaf=measured.vmaf,
                flicker=measured.flicker,
                width=info.width,
                height=info.height,
                frames=frames,
                fps_num=info.fps_num,
                fps_den=info.fps_den,
            )
        )
    return scores


def merge_bitrate(
    uploaded: Path,
    dest: Path,
    *,
    config_path: Path | None = None,
) -> list[ResultRow]:
    """Union ``uploaded`` into ``dest`` by ``(clip_id, codec, crf)``.

    An existing row is kept. A conflicting row, an empty rate column, or a
    different config hash is refused.
    """
    incoming = read_results(uploaded)
    if not incoming:
        raise BitrateError(f"{uploaded} has no rows")
    _validate_bitrate_rows(incoming)
    digest = _one_config(incoming)
    if config_path is not None and config_sha256(config_path) != digest:
        raise BitrateError(f"{uploaded} does not match {config_path}")
    if dest.is_file():
        current = read_results(dest)
        if current:
            _validate_bitrate_rows(current)
            if _one_config(current) != digest:
                raise BitrateError(
                    f"{dest} was built from a different config; refusing to mix rows"
                )
        merged = {(row.clip_id, row.codec, row.crf): row for row in current}
        for row in incoming:
            key = (row.clip_id, row.codec, row.crf)
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise BitrateError(
                    f"{dest} already has a different row for {key[0]} {key[1]} crf={key[2]}"
                )
            merged[key] = row
        rows = list(merged.values())
    else:
        rows = list(incoming)
    write_results(dest, rows)
    return sort_rows(rows)


def _result_row(
    measured: DepthScore,
    *,
    clip_id: str,
    name: str,
    revision: str,
    run_id: str,
    git_sha: str,
    config_sha256: str,
    torch_version: str,
    diffusers_version: str,
    runtime_dtype: str,
    gpu_name: str,
) -> ResultRow:
    return ResultRow(
        run_id=run_id,
        git_sha=git_sha,
        config_sha256=config_sha256,
        clip_id=clip_id,
        codec=name,
        preset=BITRATE_PRESET,
        crf=measured.bit_depth,
        width=measured.width,
        height=measured.height,
        frames=measured.frames,
        fps_num=measured.fps_num,
        fps_den=measured.fps_den,
        bits=measured.bits,
        bpp=measured.bpp,
        kbps=measured.kbps,
        encode_seconds=measured.encode_seconds,
        decode_seconds=measured.decode_seconds,
        psnr_y=measured.psnr_y,
        psnr_rgb=measured.psnr_rgb,
        ssim=measured.ssim,
        vmaf=measured.vmaf,
        flicker=measured.flicker,
        lpips=None,
        peak_vram_bytes=measured.peak_vram_bytes,
        spatial_tiling=measured.spatial_tiling,
        framewise_encoding=measured.framewise_encoding,
        framewise_decoding=measured.framewise_decoding,
        retried_with_tiling=measured.retried_with_tiling,
        per_chunk_seconds=measured.per_chunk_seconds,
        chunk_count=measured.chunk_count,
        torch_version=torch_version,
        diffusers_version=diffusers_version,
        runtime_dtype=runtime_dtype,
        gpu_name=gpu_name,
        vae_revision=revision,
        latent_shape=measured.latent_shape,
        entropy_bits=measured.entropy_bits,
    )


def _validate_bitrate_rows(rows: list[ResultRow]) -> None:
    seen: set[tuple[str, str, int | None]] = set()
    for row in rows:
        key = (row.clip_id, row.codec, row.crf)
        if key in seen:
            raise BitrateError(f"duplicate row for {row.clip_id} {row.codec} crf={row.crf}")
        seen.add(key)
        if row.codec not in CHECKPOINTS:
            raise BitrateError(f"unknown VAE {row.codec}")
        if row.preset != BITRATE_PRESET:
            raise BitrateError(f"{row.clip_id} {row.codec} preset is {row.preset!r}")
        if row.crf not in BIT_DEPTHS:
            raise BitrateError(f"{row.clip_id} {row.codec} crf {row.crf} is not a bit depth")
        if row.bits is None or row.bits < 1 or row.bpp is None or row.kbps is None:
            raise BitrateError(f"{row.clip_id} {row.codec} crf={row.crf} is missing a rate")
        expected = row.bits / (row.width * row.height * row.frames)
        if abs(row.bpp - expected) > 1e-12:
            raise BitrateError(f"{row.clip_id} {row.codec} bpp is not bits over pixels")
        if row.entropy_bits is None:
            raise BitrateError(f"{row.clip_id} {row.codec} is missing entropy_bits")
        if row.lpips is not None:
            raise BitrateError(
                f"{row.clip_id} {row.codec} filled lpips; this pass does not score it"
            )
        if row.codec == "hunyuan" and row.spatial_tiling is not True:
            raise BitrateError(f"{row.clip_id} hunyuan must have spatial tiling on")
        if not isinstance(row.spatial_tiling, bool):
            raise BitrateError(f"{row.clip_id} {row.codec} is missing spatial_tiling")
        if row.peak_vram_bytes is None:
            raise BitrateError(f"{row.clip_id} {row.codec} is missing peak_vram_bytes")
        if row.chunk_count is None or row.chunk_count < 1 or row.per_chunk_seconds is None:
            raise BitrateError(f"{row.clip_id} {row.codec} is missing per-chunk time")


def _one_config(rows: list[ResultRow]) -> str:
    digests = {row.config_sha256 for row in rows}
    if len(digests) != 1:
        raise BitrateError("rows use more than one config hash")
    return next(iter(digests))


def _encode_latent(
    vae: VideoVae, video: Any, torch: Any
) -> tuple[float, Any, int | None, float | None, bool, int | None]:
    retried = False
    for attempt in (1, 2):
        try:
            with _EncoderTimer(vae.module, torch) as timer:
                _sync(torch)
                if bool(torch.cuda.is_available()):
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                latents = vae.encode(video)
                _sync(torch)
                encode_seconds = time.perf_counter() - started
            return (
                encode_seconds,
                latents,
                timer.count,
                timer.mean_seconds,
                retried,
                _peak_bytes(torch),
            )
        except Exception as exc:
            if attempt == 1 and _is_oom(exc):
                _release_cuda()
                from latentcodecbench.vaes.smoke import enable_memory_tiling

                enable_memory_tiling(vae.module)
                retried = True
                continue
            raise
    raise BitrateError("encode retry did not return")


def _decode_restored(
    vae: VideoVae,
    restored: npt.NDArray[np.float64],
    torch: Any,
    parameter: Any,
) -> tuple[float, Any, int | None, bool]:
    tensor = torch.from_numpy(np.ascontiguousarray(restored)[None, ...]).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    retried = False
    for attempt in (1, 2):
        try:
            _sync(torch)
            if bool(torch.cuda.is_available()):
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            decoded = vae.decode(tensor)
            _sync(torch)
            return time.perf_counter() - started, decoded, _peak_bytes(torch), retried
        except Exception as exc:
            if attempt == 1 and _is_oom(exc):
                _release_cuda()
                from latentcodecbench.vaes.smoke import enable_memory_tiling

                enable_memory_tiling(vae.module)
                retried = True
                continue
            raise
    raise BitrateError("decode retry did not return")


def _higher(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _latent_numpy(latents: Any) -> npt.NDArray[np.float64]:
    array = latents.detach().float().cpu().numpy()
    if array.ndim != 5 or int(array.shape[0]) != 1:
        got = tuple(int(v) for v in array.shape)
        raise BitrateError(f"latent must be (1, channels, frames, height, width), got {got}")
    moved: npt.NDArray[np.float64] = array[0].astype(np.float64)
    return moved


def _flags(vae: VideoVae) -> dict[str, bool]:
    from latentcodecbench.vaes.smoke import tiling_flags

    return tiling_flags(vae.module)
