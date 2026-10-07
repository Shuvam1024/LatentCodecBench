"""Load the clip manifest.

The YAML file is the list of sources. Video bytes stay out of git; this
file records where they come from, the license, the native geometry, and
the SHA256 of the bytes ``lcb data fetch`` is willing to keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from latentcodecbench.data.canonical import (
    CANONICAL_FRAMES,
    CANONICAL_HEIGHT,
    CANONICAL_PIX_FMT,
    CANONICAL_WIDTH,
)

_REQUIRED = (
    "id",
    "source_url",
    "sha256",
    "license",
    "license_url",
    "native_width",
    "native_height",
    "native_fps_num",
    "native_fps_den",
    "category",
)


@dataclass(frozen=True)
class ClipSpec:
    """One source clip, before canonical preprocessing."""

    id: str
    source_url: str
    sha256: str
    license: str
    license_url: str
    native_width: int
    native_height: int
    native_fps_num: int
    native_fps_den: int
    category: str
    note: str = ""
    byte_start: int = 0
    byte_length: int | None = None
    start_frame: int = 0
    archive: str | None = None
    inner_path: str | None = None

    @property
    def native_fps(self) -> float:
        return self.native_fps_num / self.native_fps_den


@dataclass(frozen=True)
class Manifest:
    """The checked-in clip list plus the canonical geometry it targets."""

    version: int
    clips: tuple[ClipSpec, ...]

    def by_id(self, clip_id: str) -> ClipSpec:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        raise KeyError(clip_id)


def load_manifest(path: Path) -> Manifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    version = raw.get("version")
    if version != 1:
        raise ValueError(f"{path} has version {version!r}, want 1")
    canonical = raw.get("canonical")
    if not isinstance(canonical, dict):
        raise ValueError(f"{path} is missing canonical")
    if (
        canonical.get("width") != CANONICAL_WIDTH
        or canonical.get("height") != CANONICAL_HEIGHT
        or canonical.get("num_frames") != CANONICAL_FRAMES
        or canonical.get("pix_fmt") != CANONICAL_PIX_FMT
    ):
        raise ValueError(
            f"{path} canonical block must be "
            f"{CANONICAL_WIDTH}x{CANONICAL_HEIGHT}, {CANONICAL_FRAMES} frames, {CANONICAL_PIX_FMT}"
        )
    clips_raw = raw.get("clips")
    if not isinstance(clips_raw, list) or not clips_raw:
        raise ValueError(f"{path} has no clips")
    clips = tuple(_parse_clip(entry, path) for entry in clips_raw)
    ids = [clip.id for clip in clips]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has duplicate clip ids")
    return Manifest(version=version, clips=clips)


def _parse_clip(entry: object, path: Path) -> ClipSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"{path} clip entry is not a mapping")
    missing = [key for key in _REQUIRED if key not in entry]
    if missing:
        raise ValueError(f"{path} clip is missing {missing}")
    clip_id = str(entry["id"])
    if not clip_id.replace("_", "").isalnum() or clip_id != clip_id.lower():
        raise ValueError(f"clip id {clip_id!r} must be lowercase letters, digits, and underscores")
    sha = str(entry["sha256"]).lower()
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError(f"{clip_id} sha256 must be 64 hex characters")
    byte_length = entry.get("byte_length")
    if byte_length is not None:
        byte_length = int(byte_length)
        if byte_length < 1:
            raise ValueError(f"{clip_id} byte_length must be positive")
    archive = entry.get("archive")
    if archive is not None and archive not in {"7z"}:
        raise ValueError(f"{clip_id} archive {archive!r} is not supported")
    return ClipSpec(
        id=clip_id,
        source_url=str(entry["source_url"]),
        sha256=sha,
        license=str(entry["license"]),
        license_url=str(entry["license_url"]),
        native_width=int(entry["native_width"]),
        native_height=int(entry["native_height"]),
        native_fps_num=int(entry["native_fps_num"]),
        native_fps_den=int(entry["native_fps_den"]),
        category=str(entry["category"]),
        note=str(entry.get("note", "")),
        byte_start=int(entry.get("byte_start", 0)),
        byte_length=byte_length,
        start_frame=int(entry.get("start_frame", 0)),
        archive=None if archive is None else str(archive),
        inner_path=None if entry.get("inner_path") is None else str(entry["inner_path"]),
    )
