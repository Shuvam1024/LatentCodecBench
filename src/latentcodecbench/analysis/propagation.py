"""Artifact-propagation table and the L4 time estimate.

One row is one clip, one VAE, and one degraded source. The resume key is
``(clip_id, vae, source_codec, source_crf)``. The CSV is rewritten on
local disk after every row. A caller that also keeps a Drive copy passes
``push``; this module does not mount anything.

The time estimate is ``clips * (encode + variants * (encode + decode))``
with the mean encode and decode from the L4 reconstruction sweep. The
clean reconstruction is not decoded. There are nine degraded sources per
clip (three codecs, three CRF points). ADR-0012.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Protocol

import numpy as np
import numpy.typing as npt

from latentcodecbench.analysis.run import git_sha
from latentcodecbench.analysis.table import read_results
from latentcodecbench.codecs.baseline import BENCHMARK_PRESET
from latentcodecbench.codecs.degrade import degraded_path, source_clips, study_points
from latentcodecbench.data.y4m import parse_header, read_y4m
from latentcodecbench.metrics.artifacts import latent_distance, propagation_quality
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

PROPAGATION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "git_sha",
    "config_sha256",
    "clip_id",
    "vae",
    "source_codec",
    "source_crf",
    "preset",
    "width",
    "height",
    "frames",
    "latent_mse",
    "latent_relative_l2",
    "latent_cosine",
    "psnr_recon_clean",
    "psnr_recon_degraded",
    "psnr_degraded_clean",
    "blocking_clean",
    "blocking_degraded",
    "blocking_recon",
    "blocking_retention",
    "ringing_clean",
    "ringing_degraded",
    "ringing_recon",
    "ringing_retention",
    "clean_encode_seconds",
    "encode_seconds",
    "decode_seconds",
    "runtime_dtype",
    "gpu_name",
)

_OPTIONAL = frozenset({"blocking_retention", "ringing_retention"})
_CANONICAL_CLIPS = 34


class PropagationError(RuntimeError):
    """The propagation table could not be extended."""


class LatentCodec(Protocol):
    """Encode a y4m to a latent, and decode a latent back to luma."""

    def encode(self, y4m: Path) -> tuple[npt.NDArray[np.generic], float]:
        """Posterior mean and encode seconds."""

    def decode_luma(self, latent: npt.NDArray[np.generic]) -> tuple[npt.NDArray[np.uint8], float]:
        """``(frames, height, width)`` uint8 luma and decode seconds."""


@dataclass(frozen=True)
class PropagationRow:
    """One degraded source through one VAE."""

    run_id: str
    git_sha: str
    config_sha256: str
    clip_id: str
    vae: str
    source_codec: str
    source_crf: int
    preset: str
    width: int
    height: int
    frames: int
    latent_mse: float
    latent_relative_l2: float
    latent_cosine: float
    psnr_recon_clean: float
    psnr_recon_degraded: float
    psnr_degraded_clean: float
    blocking_clean: float
    blocking_degraded: float
    blocking_recon: float
    blocking_retention: float | None
    ringing_clean: float
    ringing_degraded: float
    ringing_recon: float
    ringing_retention: float | None
    clean_encode_seconds: float
    encode_seconds: float
    decode_seconds: float
    runtime_dtype: str
    gpu_name: str


@dataclass(frozen=True)
class GpuEstimate:
    """L4 wall clock for one VAE on the study ladder."""

    vae: str
    clips: int
    variants: int
    encode_mean_seconds: float
    decode_mean_seconds: float
    seconds: float


def study_variant_count() -> int:
    """Degraded sources per clip. Three codecs times three CRF points."""
    return len(study_points())


def propagation_config_sha(points: tuple[tuple[str, int], ...] | None = None) -> str:
    """SHA256 of the ladder this table is allowed to hold."""
    chosen = study_points() if points is None else points
    payload = {
        "points": [[codec, crf] for codec, crf in chosen],
        "presets": {codec: BENCHMARK_PRESET[codec] for codec, _crf in chosen},
    }
    blob = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def propagation_gpu_seconds(
    encode_mean: float,
    decode_mean: float,
    *,
    clips: int = _CANONICAL_CLIPS,
    variants: int | None = None,
) -> float:
    """Seconds for one VAE: one clean encode, then each variant encode and decode.

    The clean reconstruction is not decoded. ``variants`` defaults to the
    study ladder (nine).
    """
    count = study_variant_count() if variants is None else variants
    return clips * (encode_mean + count * (encode_mean + decode_mean))


def format_duration(seconds: float) -> str:
    """Nearest minute under 90 minutes, otherwise hours to one decimal."""
    minutes = seconds / 60.0
    if minutes < 90.0:
        return f"{round(minutes)} min"
    return f"{minutes / 60.0:.1f} h"


def l4_propagation_estimates(path: Path) -> tuple[GpuEstimate, ...]:
    """Mean L4 encode and decode from a reconstruction CSV, one row per VAE."""
    grouped: dict[str, list[float]] = defaultdict(list)
    decodes: dict[str, list[float]] = defaultdict(list)
    for row in read_results(path):
        if row.gpu_name != "NVIDIA L4" or row.runtime_dtype != "bfloat16":
            continue
        grouped[row.codec].append(row.encode_seconds)
        decodes[row.codec].append(row.decode_seconds)
    variants = study_variant_count()
    estimates: list[GpuEstimate] = []
    for name in SMOKE_ORDER:
        encodes = grouped.get(name, [])
        decode_samples = decodes.get(name, [])
        if len(encodes) != _CANONICAL_CLIPS or len(decode_samples) != _CANONICAL_CLIPS:
            raise PropagationError(
                f"{name} has {len(encodes)} L4 rows in {path}, want {_CANONICAL_CLIPS}"
            )
        encode_mean = fmean(encodes)
        decode_mean = fmean(decode_samples)
        estimates.append(
            GpuEstimate(
                vae=name,
                clips=_CANONICAL_CLIPS,
                variants=variants,
                encode_mean_seconds=encode_mean,
                decode_mean_seconds=decode_mean,
                seconds=propagation_gpu_seconds(encode_mean, decode_mean),
            )
        )
    return tuple(estimates)


def l4_runtime_markdown(path: Path) -> str:
    """The estimate table embedded in the learn note and the notebook."""
    lines = [
        "| VAE | Mean encode | Mean decode | L4 estimate |",
        "| --- | --- | --- | --- |",
    ]
    for row in l4_propagation_estimates(path):
        encode = f"{row.encode_mean_seconds:.2f} s"
        decode = f"{row.decode_mean_seconds:.2f} s"
        lines.append(f"| {row.vae} | {encode} | {decode} | {format_duration(row.seconds)} |")
    return "\n".join(lines)


def run_propagation(
    *,
    clip_dir: Path,
    degraded_dir: Path,
    out: Path,
    opener: Callable[[str], LatentCodec],
    runtime_dtype: str,
    gpu_name: str,
    push: Callable[[Path], str] | None = None,
    only: set[str] | None = None,
    vaes: tuple[str, ...] | None = None,
    points: tuple[tuple[str, int], ...] | None = None,
    now: datetime | None = None,
) -> list[PropagationRow]:
    """Score pending variants and rewrite ``out`` after each new row.

    ``push`` runs after the local rewrite. Its return value is logged.
    A push that times out does not drop the local row.
    """
    chosen = study_points() if points is None else points
    names = SMOKE_ORDER if vaes is None else vaes
    digest = propagation_config_sha(chosen)
    clips = source_clips(clip_dir, only)
    _require_degraded(clips, degraded_dir, chosen)
    _require_blocking_grid(clips)
    existing = read_propagation(out) if out.is_file() else []
    if any(row.config_sha256 != digest for row in existing):
        raise PropagationError(f"{out} was built from a different ladder; refusing to mix rows")
    done = _index(existing)
    sha = git_sha()
    if existing:
        run_id = existing[0].run_id
    else:
        moment = now if now is not None else datetime.now(UTC)
        run_id = f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{sha[:12]}-{digest[:12]}"
    rows = list(existing)
    print(f"resume {len(rows)} rows from {out}", flush=True)
    for name in names:
        session: LatentCodec | None = None
        for clip in clips:
            pending = [
                (codec, crf) for codec, crf in chosen if (clip.stem, name, codec, crf) not in done
            ]
            if not pending:
                print(f"skip {clip.stem} {name}", flush=True)
                continue
            if session is None:
                session = opener(name)
            print(f"run {clip.stem} {name}", flush=True)
            clean_latent, clean_seconds = session.encode(clip)
            _info, clean_luma, _cb, _cr = read_y4m(clip)
            for codec, crf in pending:
                variant = degraded_path(degraded_dir, clip.stem, codec, crf)
                print(f"run {clip.stem} {name} {codec} crf={crf}", flush=True)
                degraded_latent, encode_seconds = session.encode(variant)
                recon_luma, decode_seconds = session.decode_luma(degraded_latent)
                _info_d, degraded_luma, _dcb, _dcr = read_y4m(variant)
                try:
                    distance = latent_distance(clean_latent, degraded_latent)
                    quality = propagation_quality(clean_luma, degraded_luma, recon_luma)
                except ValueError as exc:
                    raise PropagationError(f"{clip.stem} {name} {codec} crf {crf}: {exc}") from exc
                rows.append(
                    PropagationRow(
                        run_id=run_id,
                        git_sha=sha,
                        config_sha256=digest,
                        clip_id=clip.stem,
                        vae=name,
                        source_codec=codec,
                        source_crf=crf,
                        preset=BENCHMARK_PRESET[codec],
                        width=int(clean_luma.shape[2]),
                        height=int(clean_luma.shape[1]),
                        frames=int(clean_luma.shape[0]),
                        latent_mse=distance.mse,
                        latent_relative_l2=distance.relative_l2,
                        latent_cosine=distance.cosine,
                        psnr_recon_clean=quality.psnr_recon_clean,
                        psnr_recon_degraded=quality.psnr_recon_degraded,
                        psnr_degraded_clean=quality.psnr_degraded_clean,
                        blocking_clean=quality.blocking_clean,
                        blocking_degraded=quality.blocking_degraded,
                        blocking_recon=quality.blocking_recon,
                        blocking_retention=quality.blocking_retention,
                        ringing_clean=quality.ringing_clean,
                        ringing_degraded=quality.ringing_degraded,
                        ringing_recon=quality.ringing_recon,
                        ringing_retention=quality.ringing_retention,
                        clean_encode_seconds=clean_seconds,
                        encode_seconds=encode_seconds,
                        decode_seconds=decode_seconds,
                        runtime_dtype=runtime_dtype,
                        gpu_name=gpu_name,
                    )
                )
                done[(clip.stem, name, codec, crf)] = rows[-1]
                write_propagation(out, rows)
                if push is not None:
                    status = push(out)
                    print(f"push {status}", flush=True)
    return rows


def write_propagation(path: Path, rows: list[PropagationRow]) -> None:
    """Rewrite ``path`` via a sibling ``*.partial`` file, then flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (row.clip_id, row.vae, row.source_codec, row.source_crf),
    )
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROPAGATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(_to_dict(row))
        handle.flush()
        with suppress(OSError):
            os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_propagation(path: Path) -> list[PropagationRow]:
    """Load a propagation CSV. An empty file yields an empty list."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = [name for name in PROPAGATION_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise PropagationError(f"{path} is missing columns {missing}")
        return [_from_dict(raw) for raw in reader]


def _require_degraded(
    clips: list[Path],
    degraded_dir: Path,
    points: tuple[tuple[str, int], ...],
) -> None:
    missing: list[Path] = []
    for clip in clips:
        for codec, crf in points:
            path = degraded_path(degraded_dir, clip.stem, codec, crf)
            if not path.is_file():
                missing.append(path)
    if missing:
        raise PropagationError(f"missing {len(missing)} degraded sources, first {missing[0]}")


def _require_blocking_grid(clips: list[Path]) -> None:
    for clip in clips:
        with clip.open("rb") as handle:
            header = parse_header(handle.read(4096))
        if header.height % 8 or header.width % 8:
            raise PropagationError(
                f"{clip.name} is {header.width}x{header.height}; blocking needs multiples of 8"
            )
        if header.height < 16 or header.width < 16:
            raise PropagationError(f"{clip.name} is smaller than two 8x8 blocks")


def propagation_key(row: PropagationRow) -> tuple[str, str, str, int]:
    """Resume identity: clip, VAE, source codec, source CRF."""
    return (row.clip_id, row.vae, row.source_codec, row.source_crf)


def merge_propagation(
    local: list[PropagationRow],
    remote: list[PropagationRow],
) -> list[PropagationRow]:
    """Union of ``local`` and ``remote``, one row per :func:`propagation_key`.

    A key present on both sides keeps the local row. Duplicate keys inside
    one side are an error. Two non-empty sides whose config hashes differ
    are an error, and the caller leaves both files alone.
    """
    local_index = _index(local)
    remote_index = _index(remote)
    local_configs = {row.config_sha256 for row in local}
    remote_configs = {row.config_sha256 for row in remote}
    if len(local_configs) > 1 or len(remote_configs) > 1:
        raise PropagationError("CSV uses more than one config")
    if local_configs and remote_configs and local_configs != remote_configs:
        raise PropagationError("local and Drive CSVs use different configs; refusing to mix rows")
    merged = dict(remote_index)
    merged.update(local_index)
    return list(merged.values())


def resume_local(local_path: Path, remote_path: Path | None) -> tuple[str, int]:
    """Merge a downloaded Drive CSV into ``local_path``.

    Returns ``(status, row_count)``. ``status`` is ``local-only``,
    ``remote-only``, or ``merged``. A config mismatch raises before
    ``local_path`` is rewritten.
    """
    local_rows = read_propagation(local_path) if local_path.is_file() else []
    remote_missing = (
        remote_path is None or not remote_path.is_file() or remote_path.stat().st_size == 0
    )
    if remote_missing:
        return "local-only", len(local_rows)
    assert remote_path is not None
    remote_rows = read_propagation(remote_path)
    if not remote_rows:
        return "local-only", len(local_rows)
    if not local_rows:
        write_propagation(local_path, remote_rows)
        return "remote-only", len(remote_rows)
    merged = merge_propagation(local_rows, remote_rows)
    write_propagation(local_path, merged)
    return "merged", len(merged)


def _index(rows: list[PropagationRow]) -> dict[tuple[str, str, str, int], PropagationRow]:
    found: dict[tuple[str, str, str, int], PropagationRow] = {}
    for row in rows:
        key = propagation_key(row)
        if key in found:
            raise PropagationError(f"duplicate row {key}")
        found[key] = row
    return found


def _to_dict(row: PropagationRow) -> dict[str, str]:
    payload: dict[str, str] = {}
    for name in PROPAGATION_COLUMNS:
        value = getattr(row, name)
        if name in _OPTIONAL:
            payload[name] = "" if value is None else repr(value)
        elif isinstance(value, float):
            payload[name] = repr(value)
        else:
            payload[name] = str(value)
    return payload


def _from_dict(raw: dict[str, str]) -> PropagationRow:
    def required(name: str) -> str:
        text = raw[name].strip()
        if text == "":
            raise PropagationError(f"empty {name}")
        return text

    def number(name: str) -> float:
        return float(required(name))

    def optional(name: str) -> float | None:
        text = raw[name].strip()
        if text == "":
            return None
        return float(text)

    return PropagationRow(
        run_id=required("run_id"),
        git_sha=required("git_sha"),
        config_sha256=required("config_sha256"),
        clip_id=required("clip_id"),
        vae=required("vae"),
        source_codec=required("source_codec"),
        source_crf=int(required("source_crf")),
        preset=required("preset"),
        width=int(required("width")),
        height=int(required("height")),
        frames=int(required("frames")),
        latent_mse=number("latent_mse"),
        latent_relative_l2=number("latent_relative_l2"),
        latent_cosine=number("latent_cosine"),
        psnr_recon_clean=number("psnr_recon_clean"),
        psnr_recon_degraded=number("psnr_recon_degraded"),
        psnr_degraded_clean=number("psnr_degraded_clean"),
        blocking_clean=number("blocking_clean"),
        blocking_degraded=number("blocking_degraded"),
        blocking_recon=number("blocking_recon"),
        blocking_retention=optional("blocking_retention"),
        ringing_clean=number("ringing_clean"),
        ringing_degraded=number("ringing_degraded"),
        ringing_recon=number("ringing_recon"),
        ringing_retention=optional("ringing_retention"),
        clean_encode_seconds=number("clean_encode_seconds"),
        encode_seconds=number("encode_seconds"),
        decode_seconds=number("decode_seconds"),
        runtime_dtype=required("runtime_dtype"),
        gpu_name=required("gpu_name"),
    )
