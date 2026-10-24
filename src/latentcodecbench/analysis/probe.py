"""Feature table for the linear probe of source CRF.

One row is one clip, one VAE, and one source. The source is the clean
clip or one degraded study point. The resume key is
``(clip_id, vae, source_codec, source_crf)``. A clean row uses the codec
name ``clean`` and an empty CRF. The CSV is rewritten on local disk
after every row. Encode only: the posterior mean is not decoded.

``source_sha256`` is the SHA256 of the y4m that was encoded. x265 and
SVT-AV1 have produced different bytes for the same settings on a new
VM, so the hash is the identity of those pixels. A file already on disk
with the right geometry is kept. ADR-0013.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from latentcodecbench.analysis.probe_features import FEATURE_COLUMNS, probe_feature_vector
from latentcodecbench.analysis.run import git_sha
from latentcodecbench.codecs.baseline import BENCHMARK_PRESET
from latentcodecbench.codecs.degrade import degraded_path, source_clips, study_points
from latentcodecbench.data.y4m import parse_header, read_y4m
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

PROBE_CSV_NAME = "latent-probe-features.csv"
CLEAN_CODEC = "clean"

PROBE_META_COLUMNS: tuple[str, ...] = (
    "run_id",
    "git_sha",
    "config_sha256",
    "clip_id",
    "vae",
    "source_codec",
    "source_crf",
    "preset",
    "source_sha256",
    "width",
    "height",
    "frames",
    "latent_shape",
    "encode_seconds",
    "runtime_dtype",
    "gpu_name",
)
PROBE_COLUMNS: tuple[str, ...] = PROBE_META_COLUMNS + FEATURE_COLUMNS


class ProbeError(RuntimeError):
    """The feature table could not be read or extended."""


class ProbeEncoder(Protocol):
    """The encode half of a loaded VAE. Decode is not part of this sweep."""

    def encode(self, y4m: Path) -> tuple[npt.NDArray[np.generic], float]:
        """Posterior mean and the encode time in seconds."""


@dataclass(frozen=True)
class ProbeRow:
    """One encoded source and its fixed-length features."""

    run_id: str
    git_sha: str
    config_sha256: str
    clip_id: str
    vae: str
    source_codec: str
    source_crf: int | None
    preset: str
    source_sha256: str
    width: int
    height: int
    frames: int
    latent_shape: str
    encode_seconds: float
    runtime_dtype: str
    gpu_name: str
    features: tuple[float, ...]


def probe_config_sha(points: tuple[tuple[str, int], ...]) -> str:
    """Hash of the ladder, the presets, and the feature names.

    A later schema does not mix into an older CSV. The hash is of those
    choices, not of the pixels.
    """
    payload = {
        "schema": 1,
        "points": points,
        "presets": {codec: BENCHMARK_PRESET[codec] for codec, _crf in points},
        "features": FEATURE_COLUMNS,
    }
    blob = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA256 of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def probe_key(row: ProbeRow) -> tuple[str, str, str, int | None]:
    """Resume identity: clip, VAE, source codec, source CRF."""
    return (row.clip_id, row.vae, row.source_codec, row.source_crf)


def run_probe(
    *,
    clip_dir: Path,
    degraded_dir: Path,
    out: Path,
    opener: Callable[[str], ProbeEncoder],
    runtime_dtype: str,
    gpu_name: str,
    push: Callable[[Path], str] | None = None,
    only: set[str] | None = None,
    vaes: tuple[str, ...] | None = None,
    points: tuple[tuple[str, int], ...] | None = None,
    now: datetime | None = None,
) -> list[ProbeRow]:
    """Encode pending sources and rewrite ``out`` after each new row.

    The clean clip is encoded once per clip and VAE that still has a
    missing row. Its latent is the reference for every new degraded row
    of that pair. A clean row already on disk is not rewritten. ``push``
    runs after the local rewrite. A push that fails does not drop the
    local row.
    """
    chosen = study_points() if points is None else points
    names = SMOKE_ORDER if vaes is None else vaes
    digest = probe_config_sha(chosen)
    clips = source_clips(clip_dir, only)
    _require_degraded(clips, degraded_dir, chosen)
    _require_blocking_grid(clips)
    existing = read_probe(out) if out.is_file() else []
    if any(row.config_sha256 != digest for row in existing):
        raise ProbeError(f"{out} was built from a different feature schema; refusing to mix rows")
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
        session: ProbeEncoder | None = None
        for clip in clips:
            pending = [
                (codec, crf) for codec, crf in chosen if (clip.stem, name, codec, crf) not in done
            ]
            need_clean = (clip.stem, name, CLEAN_CODEC, None) not in done
            if not pending and not need_clean:
                print(f"skip {clip.stem} {name}", flush=True)
                continue
            if session is None:
                session = opener(name)
            print(f"run {clip.stem} {name} clean", flush=True)
            clean_latent, clean_seconds = session.encode(clip)
            _info, clean_luma, _cb, _cr = read_y4m(clip)
            if need_clean:
                try:
                    rows.append(
                        _make_row(
                            run_id=run_id,
                            sha=sha,
                            digest=digest,
                            clip_id=clip.stem,
                            vae=name,
                            source_codec=CLEAN_CODEC,
                            source_crf=None,
                            preset="",
                            source=clip,
                            latent=clean_latent,
                            clean_latent=clean_latent,
                            luma=clean_luma,
                            encode_seconds=clean_seconds,
                            runtime_dtype=runtime_dtype,
                            gpu_name=gpu_name,
                        )
                    )
                except ValueError as exc:
                    raise ProbeError(f"{clip.stem} {name} clean: {exc}") from exc
                done[probe_key(rows[-1])] = rows[-1]
                _commit(out, rows, push)
            for codec, crf in pending:
                variant = degraded_path(degraded_dir, clip.stem, codec, crf)
                print(f"run {clip.stem} {name} {codec} crf={crf}", flush=True)
                degraded_latent, encode_seconds = session.encode(variant)
                _info_d, degraded_luma, _dcb, _dcr = read_y4m(variant)
                try:
                    rows.append(
                        _make_row(
                            run_id=run_id,
                            sha=sha,
                            digest=digest,
                            clip_id=clip.stem,
                            vae=name,
                            source_codec=codec,
                            source_crf=crf,
                            preset=BENCHMARK_PRESET[codec],
                            source=variant,
                            latent=degraded_latent,
                            clean_latent=clean_latent,
                            luma=degraded_luma,
                            encode_seconds=encode_seconds,
                            runtime_dtype=runtime_dtype,
                            gpu_name=gpu_name,
                        )
                    )
                except ValueError as exc:
                    raise ProbeError(f"{clip.stem} {name} {codec} crf {crf}: {exc}") from exc
                done[probe_key(rows[-1])] = rows[-1]
                _commit(out, rows, push)
    return rows


def write_probe(path: Path, rows: list[ProbeRow]) -> None:
    """Rewrite ``path`` via a sibling ``*.partial`` file, then flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=_sort_key)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(_to_dict(row))
        handle.flush()
        with suppress(OSError):
            os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_probe(path: Path) -> list[ProbeRow]:
    """Load a probe-feature CSV. An empty file yields an empty list."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = [name for name in PROBE_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ProbeError(f"{path} is missing columns {missing[:3]} ({len(missing)} total)")
        rows = [_from_dict(raw) for raw in reader]
    _index(rows)
    return rows


def merge_probe(local: list[ProbeRow], remote: list[ProbeRow]) -> list[ProbeRow]:
    """Union of ``local`` and ``remote``, one row per :func:`probe_key`.

    A key present on both sides keeps the local row. Duplicate keys inside
    one side are an error. Two non-empty sides whose config hashes differ
    are an error.
    """
    local_index = _index(local)
    remote_index = _index(remote)
    local_configs = {row.config_sha256 for row in local}
    remote_configs = {row.config_sha256 for row in remote}
    if len(local_configs) > 1 or len(remote_configs) > 1:
        raise ProbeError("CSV uses more than one config")
    if local_configs and remote_configs and local_configs != remote_configs:
        raise ProbeError("local and Drive CSVs use different configs; refusing to mix rows")
    merged = dict(remote_index)
    merged.update(local_index)
    return list(merged.values())


def resume_local(local_path: Path, remote_path: Path | None) -> tuple[str, int]:
    """Merge a downloaded Drive CSV into ``local_path``.

    Returns ``(status, row_count)``. ``status`` is ``local-only``,
    ``remote-only``, or ``merged``. A config mismatch raises before
    ``local_path`` is rewritten.
    """
    local_rows = read_probe(local_path) if local_path.is_file() else []
    remote_missing = (
        remote_path is None or not remote_path.is_file() or remote_path.stat().st_size == 0
    )
    if remote_missing:
        return "local-only", len(local_rows)
    assert remote_path is not None
    remote_rows = read_probe(remote_path)
    if not remote_rows:
        return "local-only", len(local_rows)
    if not local_rows:
        write_probe(local_path, remote_rows)
        return "remote-only", len(remote_rows)
    merged = merge_probe(local_rows, remote_rows)
    write_probe(local_path, merged)
    return "merged", len(merged)


def reconcile_with_drive(local_path: Path, remote_path: Path) -> str:
    """Merge the Drive copy into ``local_path``, then check the row counts.

    Returns ``ok`` when the merged local file holds at least every Drive
    row. Returns ``refused-config`` when the two files were built from
    different schemas, and ``refused-fewer-rows`` when the local file
    would still be shorter than Drive. Only ``ok`` may be uploaded.
    """
    remote_rows = read_probe(remote_path) if remote_path.is_file() else []
    try:
        _status, count = resume_local(local_path, remote_path)
    except ProbeError as exc:
        print(f"reconcile: {exc}", flush=True)
        return "refused-config"
    if count < len(remote_rows):
        print(f"reconcile: local {count} rows < Drive {len(remote_rows)} rows", flush=True)
        return "refused-fewer-rows"
    return "ok"


def _make_row(
    *,
    run_id: str,
    sha: str,
    digest: str,
    clip_id: str,
    vae: str,
    source_codec: str,
    source_crf: int | None,
    preset: str,
    source: Path,
    latent: npt.NDArray[np.generic],
    clean_latent: npt.NDArray[np.generic],
    luma: npt.NDArray[np.generic],
    encode_seconds: float,
    runtime_dtype: str,
    gpu_name: str,
) -> ProbeRow:
    vector = probe_feature_vector(latent, clean_latent, luma)
    if vector.shape != (len(FEATURE_COLUMNS),):
        raise ProbeError(f"feature length {vector.shape[0]} is not {len(FEATURE_COLUMNS)}")
    volume = latent
    if volume.ndim == 5:
        volume = volume[0]
    channels, time, height, width = (int(axis) for axis in volume.shape)
    return ProbeRow(
        run_id=run_id,
        git_sha=sha,
        config_sha256=digest,
        clip_id=clip_id,
        vae=vae,
        source_codec=source_codec,
        source_crf=source_crf,
        preset=preset,
        source_sha256=file_sha256(source),
        width=int(luma.shape[2]),
        height=int(luma.shape[1]),
        frames=int(luma.shape[0]),
        latent_shape=f"{channels} x {time} x {height} x {width}",
        encode_seconds=encode_seconds,
        runtime_dtype=runtime_dtype,
        gpu_name=gpu_name,
        features=tuple(float(value) for value in vector.tolist()),
    )


def _commit(path: Path, rows: list[ProbeRow], push: Callable[[Path], str] | None) -> None:
    write_probe(path, rows)
    if push is None:
        return
    status = push(path)
    print(f"push {status}", flush=True)


def _sort_key(row: ProbeRow) -> tuple[str, int, str, int]:
    vae_order = SMOKE_ORDER.index(row.vae) if row.vae in SMOKE_ORDER else len(SMOKE_ORDER)
    crf = -1 if row.source_crf is None else row.source_crf
    codec = "" if row.source_codec == CLEAN_CODEC else row.source_codec
    return (row.clip_id, vae_order, codec, crf)


def _index(rows: list[ProbeRow]) -> dict[tuple[str, str, str, int | None], ProbeRow]:
    found: dict[tuple[str, str, str, int | None], ProbeRow] = {}
    for row in rows:
        key = probe_key(row)
        if key in found:
            raise ProbeError(f"duplicate row {key}")
        found[key] = row
    return found


def _to_dict(row: ProbeRow) -> dict[str, str]:
    payload = {
        "run_id": row.run_id,
        "git_sha": row.git_sha,
        "config_sha256": row.config_sha256,
        "clip_id": row.clip_id,
        "vae": row.vae,
        "source_codec": row.source_codec,
        "source_crf": "" if row.source_crf is None else str(row.source_crf),
        "preset": row.preset,
        "source_sha256": row.source_sha256,
        "width": str(row.width),
        "height": str(row.height),
        "frames": str(row.frames),
        "latent_shape": row.latent_shape,
        "encode_seconds": repr(row.encode_seconds),
        "runtime_dtype": row.runtime_dtype,
        "gpu_name": row.gpu_name,
    }
    for name, value in zip(FEATURE_COLUMNS, row.features, strict=True):
        payload[name] = repr(value)
    return payload


def _from_dict(raw: dict[str, str]) -> ProbeRow:
    def required(name: str) -> str:
        text = raw[name].strip()
        if text == "":
            raise ProbeError(f"empty {name}")
        return text

    crf_text = raw["source_crf"].strip()
    codec = required("source_codec")
    if codec == CLEAN_CODEC:
        if crf_text != "":
            raise ProbeError("a clean row has a CRF")
        crf: int | None = None
    else:
        if crf_text == "":
            raise ProbeError(f"empty source_crf for {codec}")
        crf = int(crf_text)
    features = tuple(float(required(name)) for name in FEATURE_COLUMNS)
    return ProbeRow(
        run_id=required("run_id"),
        git_sha=required("git_sha"),
        config_sha256=required("config_sha256"),
        clip_id=required("clip_id"),
        vae=required("vae"),
        source_codec=codec,
        source_crf=crf,
        preset=raw["preset"].strip(),
        source_sha256=required("source_sha256"),
        width=int(required("width")),
        height=int(required("height")),
        frames=int(required("frames")),
        latent_shape=required("latent_shape"),
        encode_seconds=float(required("encode_seconds")),
        runtime_dtype=required("runtime_dtype"),
        gpu_name=required("gpu_name"),
        features=features,
    )


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
        raise ProbeError(f"missing {len(missing)} degraded sources, first {missing[0]}")


def _require_blocking_grid(clips: list[Path]) -> None:
    for clip in clips:
        with clip.open("rb") as handle:
            header = parse_header(handle.read(4096))
        if header.height % 8 or header.width % 8:
            raise ProbeError(
                f"{clip.name} is {header.width}x{header.height}; blocking needs multiples of 8"
            )
        if header.height < 16 or header.width < 16:
            raise ProbeError(f"{clip.name} is smaller than two 8x8 blocks")
