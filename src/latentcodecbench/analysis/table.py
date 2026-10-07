"""The tidy results table.

One row is one codec at one CRF on one clip. Columns are fixed, and rows
are written sorted by clip id, codec name, then CRF, so a rerun diffs
cleanly. See ``docs/results/SCHEMA.md``.
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


@dataclass(frozen=True)
class ResultRow:
    """One encoded operating point."""

    run_id: str
    git_sha: str
    config_sha256: str
    clip_id: str
    codec: str
    preset: str
    crf: int
    width: int
    height: int
    frames: int
    fps_num: int
    fps_den: int
    bits: int
    bpp: float
    kbps: float
    encode_seconds: float
    decode_seconds: float
    psnr_y: float
    psnr_rgb: float
    ssim: float
    vmaf: float
    flicker: float
    lpips: float | None


def sort_rows(rows: list[ResultRow]) -> list[ResultRow]:
    """Deterministic order: clip, codec, CRF."""
    return sorted(rows, key=lambda row: (row.clip_id, row.codec, row.crf))


def write_results(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sort_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
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


def _to_dict(row: ResultRow) -> dict[str, str]:
    payload: dict[str, str] = {}
    for name in COLUMNS:
        value = getattr(row, name)
        if value is None:
            payload[name] = ""
        elif isinstance(value, float):
            payload[name] = repr(value)
        else:
            payload[name] = str(value)
    return payload


def _from_dict(raw: dict[str, str]) -> ResultRow:
    lpips = raw["lpips"]
    return ResultRow(
        run_id=raw["run_id"],
        git_sha=raw["git_sha"],
        config_sha256=raw["config_sha256"],
        clip_id=raw["clip_id"],
        codec=raw["codec"],
        preset=raw["preset"],
        crf=int(raw["crf"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        frames=int(raw["frames"]),
        fps_num=int(raw["fps_num"]),
        fps_den=int(raw["fps_den"]),
        bits=int(raw["bits"]),
        bpp=float(raw["bpp"]),
        kbps=float(raw["kbps"]),
        encode_seconds=float(raw["encode_seconds"]),
        decode_seconds=float(raw["decode_seconds"]),
        psnr_y=float(raw["psnr_y"]),
        psnr_rgb=float(raw["psnr_rgb"]),
        ssim=float(raw["ssim"]),
        vmaf=float(raw["vmaf"]),
        flicker=float(raw["flicker"]),
        lpips=None if lpips == "" else float(lpips),
    )
