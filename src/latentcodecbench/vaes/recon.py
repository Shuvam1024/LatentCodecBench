"""Reconstruction sweep: every clip through every VAE, with empty rate columns.

The metrics are the Phase 1 scores against the source y4m. Peak VRAM,
encode and decode time, per-chunk encoder time, and the tiling flags are
recorded beside them. Hunyuan spatial tiling is enabled before the first
forward. The other VAEs stay untiled unless a forward runs out of memory,
in which case tiling is turned on and the clip is tried once more.

Rows are appended to the CSV after each clip, and a ``(clip_id, vae)``
that is already in the file is skipped. A file from a different config
is refused. Each row has a wall-clock limit: a forward or a score that
does not finish is logged and the sweep moves on. See ADR-0009 and
``docs/results/SCHEMA.md``.
"""

from __future__ import annotations

import importlib
import os
import signal
import sys
import tempfile
import threading
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
from latentcodecbench.data.y4m import Y4MInfo, read_y4m, write_y4m
from latentcodecbench.metrics.color import rgb_to_ycbcr, ycbcr_to_rgb
from latentcodecbench.metrics.score import score_y4m
from latentcodecbench.vaes.checkpoints import CHECKPOINTS, SMOKE_ORDER
from latentcodecbench.vaes.smoke import enable_memory_tiling, tiling_flags
from latentcodecbench.vaes.wrapper import VideoVae, _torch

RECON_PRESET = "reconstruction"
FULL_SWEEP_ROWS = 136

Runner = Callable[[VideoVae, Path, Path], "ClipReconstruction"]


class ReconstructionError(RuntimeError):
    """The reconstruction sweep could not continue."""


class RowTimeout(ReconstructionError):
    """One row exceeded its wall-clock limit. The sweep continues."""


@dataclass(frozen=True)
class TensorReconstruction:
    """One encode/decode, before the pixels are scored as a y4m."""

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
    pixels: Any


@dataclass(frozen=True)
class ClipReconstruction:
    """One scored reconstruction, ready to become a table row."""

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


def apply_sweep_tiling(name: str, module: Any) -> None:
    """Turn Hunyuan spatial tiling on before the first forward.

    The T4 smoke ran that VAE out of memory with the switch off. The other
    models keep the class default, which is tiling off.
    """
    if name != "hunyuan":
        return
    enable = getattr(module, "enable_tiling", None)
    if not callable(enable):
        raise ReconstructionError("hunyuan has no enable_tiling")
    enable()


def load_recon_config(path: Path) -> tuple[str, ...]:
    """Return the VAE order. The file must list ``SMOKE_ORDER`` and tile Hunyuan."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReconstructionError(f"{path} must be a mapping")
    if raw.get("version") != 1 or raw.get("kind") != "vae-reconstruction":
        raise ReconstructionError(f"{path} is not a vae-reconstruction config")
    if raw.get("hunyuan_spatial_tiling") is not True:
        raise ReconstructionError(f"{path} must enable Hunyuan spatial tiling")
    names = raw.get("vaes")
    if not isinstance(names, list) or tuple(names) != SMOKE_ORDER:
        joined = ", ".join(SMOKE_ORDER)
        raise ReconstructionError(f"{path} VAE list must be {joined}")
    return SMOKE_ORDER


def y4m_to_unit_rgb(path: Path) -> tuple[Y4MInfo, npt.NDArray[np.float64]]:
    """Read a canonical y4m as RGB in ``[-1, 1]``, shape ``(frames, height, width, 3)``."""
    info, luma, cb, cr = read_y4m(path)
    rgb = ycbcr_to_rgb(luma, cb, cr)
    return info, rgb / 127.5 - 1.0


def unit_rgb_to_planes(
    rgb: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Map ``[-1, 1]`` RGB back to limited-range 4:2:0 planes."""
    coded = np.clip((rgb.astype(np.float64) + 1.0) * 127.5, 0.0, 255.0)
    return rgb_to_ycbcr(coded)


def reconstruct_tensor(vae: VideoVae, video: Any) -> TensorReconstruction:
    """Encode and decode ``video``. One tiling retry after an out-of-memory error."""
    torch = _torch()
    retried = False
    for attempt in (1, 2):
        try:
            with _EncoderTimer(vae.module, torch) as timer:
                encode_seconds, decode_seconds, latents, decoded = _forward(vae, video, torch)
        except Exception as exc:
            if attempt == 1 and _is_oom(exc):
                _release_cuda()
                enable_memory_tiling(vae.module)
                retried = True
                continue
            raise
        flags = tiling_flags(vae.module)
        return TensorReconstruction(
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
            peak_vram_bytes=_peak_bytes(torch),
            spatial_tiling=flags["spatial_tiling"],
            framewise_encoding=flags["framewise_encoding"],
            framewise_decoding=flags["framewise_decoding"],
            retried_with_tiling=retried,
            chunk_count=timer.count,
            per_chunk_seconds=timer.mean_seconds,
            latent_shape=_latent_shape(latents),
            pixels=decoded,
        )
    raise ReconstructionError("reconstruction retry did not return")


class _RowWatch:
    """Log a heartbeat and raise :class:`RowTimeout` when a row runs too long."""

    def __init__(self, label: str, timeout: float, heartbeat: float) -> None:
        self.label = label
        self.timeout = timeout
        self.heartbeat = heartbeat
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread: threading.Thread | None = None
        self._previous: Any = None

    def __enter__(self) -> _RowWatch:
        if self.heartbeat > 0:
            self._thread = threading.Thread(target=self._beat, name="row-watchdog", daemon=True)
            self._thread.start()
        if self.timeout > 0 and hasattr(signal, "setitimer"):
            self._previous = signal.signal(signal.SIGALRM, self._handle)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self.timeout > 0 and hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._previous is not None:
                signal.signal(signal.SIGALRM, self._previous)
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _handle(self, _signum: int, _frame: object) -> None:
        raise RowTimeout(f"{self.label} exceeded {self.timeout:.3g}s")

    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat):
            elapsed = time.monotonic() - self._started
            print(f"watchdog {self.label} still running t={elapsed:.0f}s", flush=True)


def _line_buffered() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(line_buffering=True)
    except (OSError, ValueError):
        return


def _is_drive(path: Path) -> bool:
    text = path.as_posix()
    return text == "/content/drive" or text.startswith("/content/drive/")


def _read_table(path: Path, *, timeout: float) -> list[ResultRow]:
    """Read a results CSV. A Drive path is copied locally first, with a limit."""
    if _is_drive(path):
        descriptor, name = tempfile.mkstemp(prefix="lcb-resume-", suffix=".csv")
        os.close(descriptor)
        local = Path(name)
        try:
            code = transfer_file(path, local, timeout=timeout)
            if code == 3:
                return []
            if code != 0:
                raise ReconstructionError(
                    f"could not read {path}; refusing to replace the rows already stored there"
                )
            if not local.is_file() or local.stat().st_size == 0:
                return []
            return read_results(local)
        except (OSError, ValueError) as exc:
            print(f"resume unreadable {path} ({exc})", flush=True)
            return []
        finally:
            local.unlink(missing_ok=True)
    if not path.is_file():
        return []
    return read_results(path)


def _prefer_table(
    primary: list[ResultRow],
    primary_path: Path,
    secondary: list[ResultRow],
    secondary_path: Path,
) -> tuple[list[ResultRow], Path]:
    if not secondary:
        return primary, primary_path
    if not primary:
        return secondary, secondary_path
    if primary[0].config_sha256 != secondary[0].config_sha256:
        raise ReconstructionError(
            f"{primary_path} and {secondary_path} were built from different configs"
        )
    if len(secondary) > len(primary):
        return secondary, secondary_path
    return primary, primary_path


def run_reconstruction(
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
    runner: Runner | None = None,
    row_timeout: float = 900.0,
    heartbeat: float = 60.0,
    publish_timeout: float = 120.0,
) -> list[ResultRow]:
    """Score each clip with each VAE and write ``out`` after every row.

    ``skip clip vae`` is a pair already stored. ``done clip vae t=..s`` is
    printed after that row has been written. A row that exceeds
    ``row_timeout`` logs ``timeout`` and the sweep continues. When ``out``
    is on Google Drive the table is written on local disk and copied across
    in a child process, so a stuck Drive call can be killed.
    """
    _line_buffered()
    manifest = load_manifest(manifest_path)
    names = load_recon_config(config_path)
    if only is not None:
        known = {clip.id for clip in manifest.clips}
        missing = only - known
        if missing:
            raise ReconstructionError(f"unknown clip id: {', '.join(sorted(missing))}")
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
        raise ReconstructionError(
            f"{source} was built from a different config; refusing to mix rows"
        )
    print(f"resume {len(existing)} rows from {source}", flush=True)
    if existing:
        run_id = existing[0].run_id
    else:
        moment = now if now is not None else datetime.now(UTC)
        run_id = f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{sha[:12]}-{digest[:12]}"
    done = {(row.clip_id, row.codec) for row in existing}
    rows = list(existing)
    active = default_runner if runner is None else runner
    for name in names:
        pending = []
        for clip in manifest.clips:
            if only is not None and clip.id not in only:
                continue
            if (clip.id, name) in done:
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
                for clip in pending:
                    print(f"timeout {clip.id} {name} after {row_timeout:.3g}s", flush=True)
                continue
            assert vae is not None
            apply_sweep_tiling(name, vae.module)
            revision = CHECKPOINTS[name].revision
            for clip in pending:
                source_y4m = clip_dir / f"{clip.id}.y4m"
                if not source_y4m.is_file():
                    raise ReconstructionError(f"missing canonical clip {source_y4m}")
                print(f"run {clip.id} {name}", flush=True)
                started = time.monotonic()
                try:
                    with _RowWatch(f"{clip.id} {name}", row_timeout, heartbeat):
                        measured = active(vae, source_y4m, work_dir)
                except RowTimeout:
                    print(f"timeout {clip.id} {name} after {row_timeout:.3g}s", flush=True)
                    _release_cuda()
                    continue
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    print(f"fail {clip.id} {name} {detail[:500]}", flush=True)
                    _release_cuda()
                    continue
                rows.append(
                    _result_row(
                        measured,
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
                done.add((clip.id, name))
                write_results(journal, rows)
                published = True
                if mirror is not None:
                    published = transfer_file(journal, mirror, timeout=publish_timeout) == 0
                if not published:
                    print(f"publish {mirror} timed out; rows stay in {journal}", flush=True)
                elapsed = time.monotonic() - started
                print(f"done {clip.id} {name} t={elapsed:.1f}s", flush=True)
        finally:
            vae = None
            _release_cuda()
    if rows:
        write_results(journal, rows)
        if mirror is not None:
            transfer_file(journal, mirror, timeout=publish_timeout)
    return sort_rows(rows)


def default_runner(vae: VideoVae, source: Path, work_dir: Path) -> ClipReconstruction:
    """The Colab path: y4m to pixels, VAE, Phase 1 metrics, delete the reconstruction."""
    torch = _torch()
    info, rgb = y4m_to_unit_rgb(source)
    frames = int(rgb.shape[0])
    parameter = next(vae.module.parameters())
    video = _unit_rgb_to_tensor(rgb, torch, dtype=parameter.dtype, device=parameter.device)
    del rgb
    measured = reconstruct_tensor(vae, video)
    del video
    scores = _score_pixels(
        source,
        info,
        frames,
        measured.pixels,
        work_dir / f"{source.stem}-{vae.geometry.name}.y4m",
    )
    return ClipReconstruction(
        encode_seconds=measured.encode_seconds,
        decode_seconds=measured.decode_seconds,
        peak_vram_bytes=measured.peak_vram_bytes,
        spatial_tiling=measured.spatial_tiling,
        framewise_encoding=measured.framewise_encoding,
        framewise_decoding=measured.framewise_decoding,
        retried_with_tiling=measured.retried_with_tiling,
        chunk_count=measured.chunk_count,
        per_chunk_seconds=measured.per_chunk_seconds,
        latent_shape=measured.latent_shape,
        psnr_y=scores.psnr_y,
        psnr_rgb=scores.psnr_rgb,
        ssim=scores.ssim,
        vmaf=scores.vmaf,
        flicker=scores.flicker,
        width=info.width,
        height=info.height,
        frames=frames,
        fps_num=info.fps_num,
        fps_den=info.fps_den,
    )


def merge_reconstruction(
    uploaded: Path,
    dest: Path,
    *,
    config_path: Path | None = None,
) -> list[ResultRow]:
    """Union ``uploaded`` into ``dest`` by ``(clip_id, codec)``.

    An existing row is kept. A second row for the same key with different
    cells is refused, and so is a file whose config hash does not match.
    """
    incoming = read_results(uploaded)
    if not incoming:
        raise ReconstructionError(f"{uploaded} has no rows")
    _validate_recon_rows(incoming)
    digest = _one_config(incoming)
    if config_path is not None and config_sha256(config_path) != digest:
        raise ReconstructionError(f"{uploaded} does not match {config_path}")
    if dest.is_file():
        current = read_results(dest)
        if current:
            _validate_recon_rows(current)
            if _one_config(current) != digest:
                raise ReconstructionError(
                    f"{dest} was built from a different config; refusing to mix rows"
                )
        merged = {(row.clip_id, row.codec): row for row in current}
        for row in incoming:
            key = (row.clip_id, row.codec)
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise ReconstructionError(
                    f"{dest} already has a different row for {key[0]} {key[1]}"
                )
            merged[key] = row
        rows = list(merged.values())
    else:
        rows = list(incoming)
    write_results(dest, rows)
    return sort_rows(rows)


def _result_row(
    measured: ClipReconstruction,
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
        preset=RECON_PRESET,
        crf=None,
        width=measured.width,
        height=measured.height,
        frames=measured.frames,
        fps_num=measured.fps_num,
        fps_den=measured.fps_den,
        bits=None,
        bpp=None,
        kbps=None,
        encode_seconds=measured.encode_seconds,
        decode_seconds=measured.decode_seconds,
        psnr_y=measured.psnr_y,
        psnr_rgb=measured.psnr_rgb,
        ssim=measured.ssim,
        vmaf=measured.vmaf,
        flicker=measured.flicker,
        # This pass does not run LPIPS. The column stays empty.
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
    )


def _validate_recon_rows(rows: list[ResultRow]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.clip_id, row.codec)
        if key in seen:
            raise ReconstructionError(f"duplicate row for {row.clip_id} {row.codec}")
        seen.add(key)
        if row.codec not in CHECKPOINTS:
            raise ReconstructionError(f"unknown VAE {row.codec}")
        if row.preset != RECON_PRESET:
            raise ReconstructionError(f"{row.clip_id} {row.codec} preset is {row.preset!r}")
        rated = (row.crf, row.bits, row.bpp, row.kbps)
        if any(value is not None for value in rated):
            raise ReconstructionError(f"{row.clip_id} {row.codec} has a rate column filled in")
        if row.codec == "hunyuan" and row.spatial_tiling is not True:
            raise ReconstructionError(f"{row.clip_id} hunyuan must have spatial tiling on")
        if not isinstance(row.spatial_tiling, bool):
            raise ReconstructionError(f"{row.clip_id} {row.codec} is missing spatial_tiling")
        framewise = (row.framewise_encoding, row.framewise_decoding)
        if not all(isinstance(flag, bool) for flag in framewise):
            raise ReconstructionError(f"{row.clip_id} {row.codec} is missing framewise flags")
        if not isinstance(row.retried_with_tiling, bool):
            raise ReconstructionError(f"{row.clip_id} {row.codec} is missing retried_with_tiling")
        if row.peak_vram_bytes is None:
            raise ReconstructionError(f"{row.clip_id} {row.codec} is missing peak_vram_bytes")
        if row.chunk_count is None or row.chunk_count < 1 or row.per_chunk_seconds is None:
            raise ReconstructionError(f"{row.clip_id} {row.codec} is missing per-chunk time")
        for field in (
            "torch_version",
            "diffusers_version",
            "runtime_dtype",
            "gpu_name",
            "vae_revision",
            "latent_shape",
        ):
            value = getattr(row, field)
            if not isinstance(value, str) or value == "":
                raise ReconstructionError(f"{row.clip_id} {row.codec} is missing {field}")


def _one_config(rows: list[ResultRow]) -> str:
    digests = {row.config_sha256 for row in rows}
    if len(digests) != 1:
        raise ReconstructionError("rows use more than one config hash")
    return next(iter(digests))


def _score_pixels(
    source: Path,
    info: Y4MInfo,
    frames: int,
    pixels: Any,
    destination: Path,
) -> Any:
    rgb = _tensor_to_unit_rgb(pixels)
    if rgb.shape != (frames, info.height, info.width, 3):
        raise ReconstructionError(
            f"decoded RGB shape {rgb.shape} != {(frames, info.height, info.width, 3)}"
        )
    luma, cb, cr = unit_rgb_to_planes(rgb)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_y4m(destination, luma, cb, cr, fps_num=info.fps_num, fps_den=info.fps_den)
    try:
        return score_y4m(source, destination)
    finally:
        destination.unlink(missing_ok=True)


def _unit_rgb_to_tensor(
    rgb: npt.NDArray[np.float64],
    torch: Any,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    # (frames, height, width, 3) -> (1, 3, frames, height, width)
    channels_first = np.transpose(rgb, (3, 0, 1, 2))
    batched = np.ascontiguousarray(channels_first)[None, ...]
    return torch.from_numpy(batched).to(device=device, dtype=dtype)


def _tensor_to_unit_rgb(tensor: Any) -> npt.NDArray[np.float64]:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim != 5 or int(array.shape[0]) != 1 or int(array.shape[1]) != 3:
        got = tuple(int(v) for v in array.shape)
        raise ReconstructionError(
            f"decoded tensor must be (1, 3, frames, height, width), got {got}"
        )
    moved: npt.NDArray[np.float64] = np.transpose(array[0], (1, 2, 3, 0)).astype(np.float64)
    return moved


def _forward(vae: VideoVae, video: Any, torch: Any) -> tuple[float, float, Any, Any]:
    _sync(torch)
    if bool(torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    latents = vae.encode(video)
    _sync(torch)
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    decoded = vae.decode(latents)
    _sync(torch)
    decode_seconds = time.perf_counter() - started
    return encode_seconds, decode_seconds, latents, decoded


def _latent_shape(latents: Any) -> str:
    dims = [int(v) for v in latents.shape[1:]]
    return "x".join(str(part) for part in dims)


class _EncoderTimer:
    """Time each ``encoder.forward`` while an encode is running."""

    def __init__(self, module: Any, torch: Any) -> None:
        self._torch = torch
        self._encoder = getattr(module, "encoder", None)
        self._original: Any = None
        forward = getattr(self._encoder, "forward", None)
        self.watched = self._encoder is not None and callable(forward)
        self.seconds: list[float] = []

    def __enter__(self) -> _EncoderTimer:
        encoder = self._encoder
        if not self.watched or encoder is None:
            return self
        original = encoder.forward
        self._original = original
        torch = self._torch
        seconds = self.seconds

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            _sync(torch)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _sync(torch)
                seconds.append(time.perf_counter() - started)

        encoder.forward = wrapped
        return self

    def __exit__(self, *_exc: object) -> None:
        encoder = self._encoder
        if self._original is not None and encoder is not None:
            encoder.forward = self._original
        return None

    @property
    def count(self) -> int | None:
        if not self.watched:
            return None
        return len(self.seconds)

    @property
    def mean_seconds(self) -> float | None:
        if not self.seconds:
            return None
        return sum(self.seconds) / len(self.seconds)


def _sync(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize()


def _peak_bytes(torch: Any) -> int | None:
    if not bool(torch.cuda.is_available()):
        return None
    return int(torch.cuda.max_memory_allocated())


def _release_cuda() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()
