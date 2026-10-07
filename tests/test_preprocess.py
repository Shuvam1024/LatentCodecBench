"""Canonical preprocessing against a ffmpeg-generated source."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from latentcodecbench.data.canonical import CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH
from latentcodecbench.data.manifest import ClipSpec
from latentcodecbench.data.preprocess import preprocess_clip
from latentcodecbench.data.y4m import read_y4m


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_preprocess_cover_crops_to_canonical(tmp_path: Path) -> None:
    import subprocess

    source = tmp_path / "source.y4m"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25",
            "-frames:v",
            "40",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    clip = ClipSpec(
        id="synthetic",
        source_url=source.as_uri(),
        sha256=digest,
        license="test",
        license_url="https://example.com/license",
        native_width=640,
        native_height=360,
        native_fps_num=25,
        native_fps_den=1,
        category="test",
        start_frame=2,
    )
    destination = preprocess_clip(clip, source, tmp_path / "clips")
    info, luma, _, _ = read_y4m(destination)
    assert (info.width, info.height) == (CANONICAL_WIDTH, CANONICAL_HEIGHT)
    assert luma.shape == (CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH)
    assert info.fps_num == 25


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("7z") is None,
    reason="ffmpeg and 7z are required",
)
def test_7z_member_is_preprocessed(tmp_path: Path) -> None:
    import subprocess

    from latentcodecbench.data.y4m import write_y4m

    luma = np.zeros((40, 16, 16), dtype=np.uint8)
    luma[:, :, :8] = 200
    chroma = np.full((40, 8, 8), 128, dtype=np.uint8)
    inner = tmp_path / "inner.y4m"
    write_y4m(inner, luma, chroma, chroma, fps_num=30, fps_den=1)
    archive = tmp_path / "inner.7z"
    subprocess.run(["7z", "a", "-y", str(archive), str(inner)], check=True, capture_output=True)
    clip = ClipSpec(
        id="packed",
        source_url=archive.as_uri(),
        sha256="ab" * 32,
        license="test",
        license_url="https://example.com/license",
        native_width=16,
        native_height=16,
        native_fps_num=30,
        native_fps_den=1,
        category="test",
        archive="7z",
        inner_path="inner.y4m",
    )
    destination = preprocess_clip(clip, archive, tmp_path / "clips")
    info, got, _, _ = read_y4m(destination)
    assert (info.width, info.height) == (CANONICAL_WIDTH, CANONICAL_HEIGHT)
    assert got.shape[0] == CANONICAL_FRAMES
