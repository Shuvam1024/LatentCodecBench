"""Run the classic-codec sweep and append a tidy table.

The run id is ``{utc}-{git_sha12}-{config_sha12}``. The git SHA is the
revision that executed the sweep, not the later commit that adds the
CSV. A table that already exists is resumed: a finished (clip, codec,
CRF) is not encoded again, and a table from a different config is refused.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from latentcodecbench.analysis.config import RunConfig, load_config
from latentcodecbench.analysis.table import ResultRow, read_results, write_results
from latentcodecbench.codecs.baseline import encode_y4m
from latentcodecbench.data.manifest import Manifest, load_manifest
from latentcodecbench.data.preprocess import ffmpeg_binary
from latentcodecbench.metrics.score import score_y4m


class RunError(RuntimeError):
    """The sweep could not continue."""


def git_sha() -> str:
    override = os.environ.get("LATENTCODECBENCH_GIT_SHA")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RunError(completed.stderr.strip() or "git rev-parse failed")
    return completed.stdout.strip()


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_baselines(
    *,
    manifest_path: Path,
    config_path: Path,
    clip_dir: Path,
    out: Path,
    work_dir: Path,
    only: set[str] | None = None,
    now: datetime | None = None,
) -> list[ResultRow]:
    """Encode, score, and write ``out``. Returns the sorted rows."""
    manifest = load_manifest(manifest_path)
    config = load_config(config_path)
    if only is not None:
        known = {clip.id for clip in manifest.clips}
        missing = only - known
        if missing:
            raise RunError(f"unknown clip id: {', '.join(sorted(missing))}")
    digest = config_sha256(config_path)
    sha = git_sha()
    existing = read_results(out) if out.is_file() else []
    if existing and existing[0].config_sha256 != digest:
        raise RunError(f"{out} was built from a different config; refusing to mix rows")
    if existing:
        run_id = existing[0].run_id
    else:
        moment = now if now is not None else datetime.now(UTC)
        run_id = f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{sha[:12]}-{digest[:12]}"
    done = {(row.clip_id, row.codec, row.crf) for row in existing}
    rows = list(existing)
    for clip in manifest.clips:
        if only is not None and clip.id not in only:
            continue
        source = clip_dir / f"{clip.id}.y4m"
        if not source.is_file():
            raise RunError(f"missing canonical clip {source}")
        for codec in config.codecs:
            for crf in codec.crf:
                key = (clip.id, codec.name, crf)
                if key in done:
                    continue
                print(f"{clip.id} {codec.name} crf {crf}", flush=True)
                rows.append(
                    _one_point(
                        source=source,
                        clip_id=clip.id,
                        codec=codec.name,
                        preset=codec.preset,
                        crf=crf,
                        work_dir=work_dir,
                        run_id=run_id,
                        git_sha=sha,
                        config_sha256=digest,
                    )
                )
                done.add(key)
                write_results(out, rows)
    write_results(out, rows)
    return rows


def _one_point(
    *,
    source: Path,
    clip_id: str,
    codec: str,
    preset: str,
    crf: int,
    work_dir: Path,
    run_id: str,
    git_sha: str,
    config_sha256: str,
) -> ResultRow:
    point_dir = work_dir / clip_id
    point_dir.mkdir(parents=True, exist_ok=True)
    bitstream = point_dir / f"{codec}-crf{crf}.mp4"
    decoded = point_dir / f"{codec}-crf{crf}.y4m"
    point = encode_y4m(source, bitstream, codec=codec, crf=crf, preset=preset)
    _decode(bitstream, decoded)
    try:
        scores = score_y4m(source, decoded)
    finally:
        decoded.unlink(missing_ok=True)
    return ResultRow(
        run_id=run_id,
        git_sha=git_sha,
        config_sha256=config_sha256,
        clip_id=clip_id,
        codec=codec,
        preset=preset,
        crf=crf,
        width=point.width,
        height=point.height,
        frames=point.frames,
        fps_num=point.fps_num,
        fps_den=point.fps_den,
        bits=point.bits,
        bpp=point.bpp,
        kbps=point.kbps,
        encode_seconds=point.encode_seconds,
        decode_seconds=point.decode_seconds,
        psnr_y=scores.psnr_y,
        psnr_rgb=scores.psnr_rgb,
        ssim=scores.ssim,
        vmaf=scores.vmaf,
        flicker=scores.flicker,
        lpips=scores.lpips,
    )


def _decode(bitstream: Path, destination: Path) -> None:
    completed = subprocess.run(
        [
            ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "1",
            "-i",
            str(bitstream),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "decode failed"
        raise RunError(detail)


def config_matches_manifest_expectation(config: RunConfig, manifest: Manifest) -> int:
    """Number of encodes a full sweep of this config would perform."""
    return len(manifest.clips) * sum(len(codec.crf) for codec in config.codecs)
