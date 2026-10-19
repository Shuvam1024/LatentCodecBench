"""The tidy results table.

One row is one codec at one CRF on one clip, or one VAE reconstruction
of one clip. The classic columns are always present. A reconstruction
leaves the rate columns empty and adds the VAE columns. Rows are written
sorted by clip id, codec name, then CRF, so a rerun diffs cleanly. An
empty CRF sorts first. See ``docs/results/SCHEMA.md``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

COLUMNS: tuple[str, ...] = (
    "run_id",
    "git_sha",
    "config_sha256",
    "clip_id",
    "codec",
    "preset",
    "crf",
    "width",
    "height",
    "frames",
    "fps_num",
    "fps_den",
    "bits",
    "bpp",
    "kbps",
    "encode_seconds",
    "decode_seconds",
    "psnr_y",
    "psnr_rgb",
    "ssim",
    "vmaf",
    "flicker",
    "lpips",
)

# Present on a reconstruction CSV. Absent on the classic table, which still loads.
VAE_COLUMNS: tuple[str, ...] = (
    "peak_vram_bytes",
    "spatial_tiling",
    "framewise_encoding",
    "framewise_decoding",
    "retried_with_tiling",
    "per_chunk_seconds",
    "chunk_count",
    "torch_version",
    "diffusers_version",
    "runtime_dtype",
    "gpu_name",
    "vae_revision",
    "latent_shape",
)


@dataclass(frozen=True)
class ResultRow:
    """One encoded operating point."""

    run_id: str
    git_sha: str
    config_sha256: str
    clip_id: str
    codec: str
    preset: str
    crf: int | None
    width: int
    height: int
    frames: int
    fps_num: int
    fps_den: int
    bits: int | None
    bpp: float | None
    kbps: float | None
    encode_seconds: float
    decode_seconds: float
    psnr_y: float
    psnr_rgb: float
    ssim: float
    vmaf: float
    flicker: float
    lpips: float | None
    peak_vram_bytes: int | None = None
    spatial_tiling: bool | None = None
    framewise_encoding: bool | None = None
    framewise_decoding: bool | None = None
    retried_with_tiling: bool | None = None
    per_chunk_seconds: float | None = None
    chunk_count: int | None = None
    torch_version: str | None = None
    diffusers_version: str | None = None
    runtime_dtype: str | None = None
    gpu_name: str | None = None
    vae_revision: str | None = None
    latent_shape: str | None = None


def sort_rows(rows: list[ResultRow]) -> list[ResultRow]:
    """Deterministic order: clip, codec, CRF. An empty CRF sorts first."""
    return sorted(rows, key=lambda row: (row.clip_id, row.codec, _crf_key(row)))


def write_results(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sort_rows(rows)
    fieldnames = COLUMNS + VAE_COLUMNS if _uses_vae_columns(ordered) else COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in ordered:
            writer.writerow(_to_dict(row))


def read_results(path: Path) -> list[ResultRow]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = [name for name in COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        return [_from_dict(raw) for raw in reader]


def _crf_key(row: ResultRow) -> int:
    return -1 if row.crf is None else row.crf


def _uses_vae_columns(rows: list[ResultRow]) -> bool:
    return any(getattr(row, name) is not None for row in rows for name in VAE_COLUMNS)


def _to_dict(row: ResultRow) -> dict[str, str]:
    payload: dict[str, str] = {}
    for name in COLUMNS + VAE_COLUMNS:
        value = getattr(row, name)
        if value is None:
            payload[name] = ""
        elif isinstance(value, bool):
            payload[name] = "1" if value else "0"
        elif isinstance(value, float):
            payload[name] = repr(value)
        else:
            payload[name] = str(value)
    return payload


def _from_dict(raw: dict[str, str]) -> ResultRow:
    return ResultRow(
        run_id=raw["run_id"],
        git_sha=raw["git_sha"],
        config_sha256=raw["config_sha256"],
        clip_id=raw["clip_id"],
        codec=raw["codec"],
        preset=raw["preset"],
        crf=_optional_int(raw["crf"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        frames=int(raw["frames"]),
        fps_num=int(raw["fps_num"]),
        fps_den=int(raw["fps_den"]),
        bits=_optional_int(raw["bits"]),
        bpp=_optional_float(raw["bpp"]),
        kbps=_optional_float(raw["kbps"]),
        encode_seconds=float(raw["encode_seconds"]),
        decode_seconds=float(raw["decode_seconds"]),
        psnr_y=float(raw["psnr_y"]),
        psnr_rgb=float(raw["psnr_rgb"]),
        ssim=float(raw["ssim"]),
        vmaf=float(raw["vmaf"]),
        flicker=float(raw["flicker"]),
        lpips=_optional_float(raw["lpips"]),
        peak_vram_bytes=_optional_int(_cell(raw, "peak_vram_bytes")),
        spatial_tiling=_optional_bool(_cell(raw, "spatial_tiling")),
        framewise_encoding=_optional_bool(_cell(raw, "framewise_encoding")),
        framewise_decoding=_optional_bool(_cell(raw, "framewise_decoding")),
        retried_with_tiling=_optional_bool(_cell(raw, "retried_with_tiling")),
        per_chunk_seconds=_optional_float(_cell(raw, "per_chunk_seconds")),
        chunk_count=_optional_int(_cell(raw, "chunk_count")),
        torch_version=_optional_text(_cell(raw, "torch_version")),
        diffusers_version=_optional_text(_cell(raw, "diffusers_version")),
        runtime_dtype=_optional_text(_cell(raw, "runtime_dtype")),
        gpu_name=_optional_text(_cell(raw, "gpu_name")),
        vae_revision=_optional_text(_cell(raw, "vae_revision")),
        latent_shape=_optional_text(_cell(raw, "latent_shape")),
    )


def _cell(raw: dict[str, str], name: str) -> str:
    value = raw.get(name)
    return "" if value is None else value


def _optional_text(text: str) -> str | None:
    return None if text == "" else text


def _optional_int(text: str) -> int | None:
    return None if text == "" else int(text)


def _optional_float(text: str) -> float | None:
    return None if text == "" else float(text)


def _optional_bool(text: str) -> bool | None:
    if text == "":
        return None
    if text in {"1", "true", "True"}:
        return True
    if text in {"0", "false", "False"}:
        return False
    raise ValueError(f"expected a flag, got {text!r}")
