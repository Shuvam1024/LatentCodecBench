"""Degraded sources for the codec-leak study.

Each canonical clip is re-encoded at three CRF points taken from the
Phase 1 ladders, then decoded back to y4m. The points are the ends of
each ladder plus the middle step. x264 and x265 use 18, 30, and 38.
SVT-AV1 uses 28, 49, and 63. Presets stay on the Phase 1 benchmark
values. ADR-0012.

Output lives under ``data/degraded/``, which ``/data/`` already
gitignores. A y4m that already matches the source geometry is left
alone, so a stopped run continues.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from latentcodecbench.codecs.baseline import (
    BENCHMARK_PRESET,
    SVT_AV1_CRF,
    X264_CRF,
    X265_CRF,
    EncodeError,
    decode_y4m,
    encode_y4m,
)
from latentcodecbench.data.y4m import parse_header

# Ends plus the middle step of each Phase 1 ladder. ADR-0005 defines the
# full six-point lists. These three are a subset, locked by a test.
STUDY_LADDER: dict[str, tuple[int, ...]] = {
    "libx264": (18, 30, 38),
    "libx265": (18, 30, 38),
    "libsvtav1": (28, 49, 63),
}

PHASE1_LADDER: dict[str, tuple[int, ...]] = {
    "libx264": X264_CRF,
    "libx265": X265_CRF,
    "libsvtav1": SVT_AV1_CRF,
}


@dataclass(frozen=True)
class DegradedClip:
    """One re-encoded source, or a file a previous run already finished."""

    clip_id: str
    codec: str
    crf: int
    preset: str
    source: Path
    y4m: Path
    width: int
    height: int
    frames: int
    encode_seconds: float
    pixel_decode_seconds: float
    skipped: bool


def study_points() -> tuple[tuple[str, int], ...]:
    """``(codec, crf)`` in ladder order."""
    return tuple((codec, crf) for codec, crfs in STUDY_LADDER.items() for crf in crfs)


def degraded_path(out_dir: Path, clip_id: str, codec: str, crf: int) -> Path:
    """Flat name ``{clip}__{codec}_crf{crf}.y4m`` under ``out_dir``."""
    return out_dir / f"{clip_id}__{codec}_crf{crf}.y4m"


def source_clips(clip_dir: Path, only: set[str] | None = None) -> list[Path]:
    """Canonical ``*.y4m`` files, optionally restricted to clip ids."""
    found = sorted(path for path in clip_dir.glob("*.y4m") if path.is_file())
    if only is not None:
        wanted = set(only)
        found = [path for path in found if path.stem in wanted]
        missing = wanted - {path.stem for path in found}
        if missing:
            joined = ", ".join(sorted(missing))
            raise EncodeError(f"missing clips: {joined}")
    if not found:
        raise EncodeError(f"no y4m files in {clip_dir}")
    return found


def degrade_sources(
    clip_dir: Path,
    out_dir: Path,
    *,
    only: set[str] | None = None,
    presets: Mapping[str, str] | None = None,
    points: tuple[tuple[str, int], ...] | None = None,
) -> list[DegradedClip]:
    """Encode every study point and decode pixels.

    ``presets`` overrides :data:`BENCHMARK_PRESET` for a smoke run. The
    study itself leaves it empty and uses medium / medium / preset 6.
    ``points`` overrides the ladder the same way. A finished y4m is skipped.
    """
    chosen_points = study_points() if points is None else points
    chosen_presets = BENCHMARK_PRESET if presets is None else presets
    written: list[DegradedClip] = []
    for source in source_clips(clip_dir, only):
        for codec, crf in chosen_points:
            if codec not in chosen_presets:
                raise EncodeError(f"no preset for {codec}")
            dest = degraded_path(out_dir, source.stem, codec, crf)
            written.append(
                degrade_one(
                    source,
                    dest,
                    codec=codec,
                    crf=crf,
                    preset=chosen_presets[codec],
                )
            )
    return written


def degrade_one(
    source: Path,
    dest: Path,
    *,
    codec: str,
    crf: int,
    preset: str,
) -> DegradedClip:
    """Encode ``source`` at one CRF and write pixels to ``dest``."""
    expected = _geometry(source)
    if dest.is_file() and _same_geometry(dest, expected):
        width, height, frames = expected[0], expected[1], expected[2]
        return DegradedClip(
            clip_id=source.stem,
            codec=codec,
            crf=crf,
            preset=preset,
            source=source,
            y4m=dest,
            width=width,
            height=height,
            frames=frames,
            encode_seconds=0.0,
            pixel_decode_seconds=0.0,
            skipped=True,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    try:
        with tempfile.TemporaryDirectory(dir=dest.parent) as temporary:
            bitstream = Path(temporary) / "bitstream.mp4"
            point = encode_y4m(
                source,
                bitstream,
                codec=codec,
                crf=crf,
                preset=preset,
                measure_decode=False,
            )
            pixel_decode_seconds = decode_y4m(bitstream, partial)
        if not _same_geometry(partial, expected):
            raise EncodeError(f"{partial.name} geometry does not match {source.name}")
        os.replace(partial, dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    width, height, frames = expected[0], expected[1], expected[2]
    return DegradedClip(
        clip_id=source.stem,
        codec=codec,
        crf=crf,
        preset=point.preset,
        source=source,
        y4m=dest,
        width=width,
        height=height,
        frames=frames,
        encode_seconds=point.encode_seconds,
        pixel_decode_seconds=pixel_decode_seconds,
        skipped=False,
    )


def _geometry(path: Path) -> tuple[int, int, int, int, int]:
    try:
        with path.open("rb") as handle:
            header = parse_header(handle.read(4096))
        payload = path.stat().st_size - header.header_length
        frames = payload // header.frame_bytes
    except (OSError, ValueError, ZeroDivisionError) as exc:
        raise EncodeError(f"{path} is not a 4:2:0 y4m") from exc
    if frames < 1 or payload % header.frame_bytes != 0:
        raise EncodeError(f"{path} is not an integer number of y4m frames")
    return header.width, header.height, frames, header.fps_num, header.fps_den


def _same_geometry(path: Path, expected: tuple[int, int, int, int, int]) -> bool:
    try:
        return _geometry(path) == expected
    except EncodeError:
        return False
