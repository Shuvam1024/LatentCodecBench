"""Turn a fetched source into the canonical y4m clip.

Scaling uses ffmpeg's lanczos filter, cover-then-center-crop, so the
picture is never stretched and never letterboxed. The source stays in
YUV; there is no trip through RGB. See ADR-0002.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from latentcodecbench.data.canonical import CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH
from latentcodecbench.data.manifest import ClipSpec
from latentcodecbench.data.y4m import read_y4m


class PreprocessError(RuntimeError):
    """ffmpeg did not produce a canonical clip."""


def ffmpeg_binary() -> str:
    override = os.environ.get("LATENTCODECBENCH_FFMPEG")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found is None:
        raise PreprocessError("ffmpeg is not on PATH")
    return found


def preprocess_clip(clip: ClipSpec, source: Path, clip_dir: Path) -> Path:
    """Write ``clip_dir / {id}.y4m`` at the canonical geometry."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    destination = clip_dir / f"{clip.id}.y4m"
    if clip.archive is None:
        _run_ffmpeg(clip, destination, input_path=source)
    elif clip.archive == "7z":
        _preprocess_7z(clip, source, destination)
    else:
        raise PreprocessError(f"{clip.id}: archive {clip.archive} is not supported")
    info, luma, _, _ = read_y4m(destination)
    if info.width != CANONICAL_WIDTH or info.height != CANONICAL_HEIGHT:
        raise PreprocessError(f"{clip.id}: got {info.width}x{info.height}")
    if luma.shape[0] != CANONICAL_FRAMES:
        raise PreprocessError(f"{clip.id}: got {luma.shape[0]} frames, want {CANONICAL_FRAMES}")
    return destination


def _filtergraph(clip: ClipSpec) -> str:
    return (
        f"select=gte(n\\,{clip.start_frame}),setpts=PTS-STARTPTS,"
        f"scale={CANONICAL_WIDTH}:{CANONICAL_HEIGHT}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={CANONICAL_WIDTH}:{CANONICAL_HEIGHT},format=yuv420p"
    )


def _ffmpeg_command(clip: ClipSpec, destination: Path, input_spec: str) -> list[str]:
    return [
        ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_spec,
        "-vf",
        _filtergraph(clip),
        "-frames:v",
        str(CANONICAL_FRAMES),
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        str(destination),
    ]


def _run_ffmpeg(clip: ClipSpec, destination: Path, *, input_path: Path) -> None:
    completed = subprocess.run(
        _ffmpeg_command(clip, destination, str(input_path)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "ffmpeg failed"
        raise PreprocessError(f"{clip.id}: {detail}")


def _preprocess_7z(clip: ClipSpec, source: Path, destination: Path) -> None:
    """Stream one archive member into ffmpeg.

    UVG ships multi-gigabyte y4m files inside 7z. ffmpeg stops after
    ``CANONICAL_FRAMES`` frames, which closes the pipe, so 7z does not have
    to unpack the rest of the member.
    """
    if not clip.inner_path:
        raise PreprocessError(f"{clip.id}: 7z clip is missing inner_path")
    seven = shutil.which("7z")
    if seven is None:
        raise PreprocessError("7z is not on PATH (install p7zip-full)")
    extractor = subprocess.Popen(
        [seven, "e", "-so", str(source), clip.inner_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert extractor.stdout is not None
    ffmpeg = subprocess.Popen(
        _ffmpeg_command(clip, destination, "pipe:0"),
        stdin=extractor.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    extractor.stdout.close()
    _, ffmpeg_err = ffmpeg.communicate()
    extractor.wait()
    if ffmpeg.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = ffmpeg_err.decode("utf-8", errors="replace").strip() or "ffmpeg failed"
        raise PreprocessError(f"{clip.id}: {detail}")
