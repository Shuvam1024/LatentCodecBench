"""ffmpeg wrappers for libx264, libx265, and libsvtav1.

CRF points are spread from a high-quality setting to a low one. The
scales are not comparable across codecs: x264 CRF 28 is not SVT-AV1 CRF
28. Rate-distortion curves, not the CRF numbers, are what get compared.

Presets and the single-thread flags are fixed so a later machine can
repeat the timing. SVT-AV1's ffmpeg default tune on this build is PSNR.
The wrapper sets ``tune=0`` (visual quality) instead, for the same reason
x264 and x265 are not passed ``-tune psnr``: the benchmark reports VMAF
as well as PSNR. See ADR-0005.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from latentcodecbench.codecs.rate import bits_per_pixel, kilobits_per_second
from latentcodecbench.data.preprocess import ffmpeg_binary
from latentcodecbench.data.y4m import parse_header

# Six points each. The ends are the quality range the three codecs
# actually share: x264 CRF 43 sits below SVT-AV1 at CRF 63, and SVT-AV1
# CRF 20 sits above x264 CRF 18. Interior points are an even step across
# that span. See ADR-0005.
X264_CRF: tuple[int, ...] = (18, 22, 26, 30, 34, 38)
X265_CRF: tuple[int, ...] = (18, 22, 26, 30, 34, 38)
SVT_AV1_CRF: tuple[int, ...] = (28, 35, 42, 49, 56, 63)

# Timed on crowd_run (832x480, 33 frames, one thread): x264 medium 0.7 s,
# x265 medium 1.5 s, SVT-AV1 preset 6 about 2 s. Preset 8 was 1.2 s.
# Preset 6 is the slower of those two and the one the sweep uses.
BENCHMARK_PRESET: dict[str, str] = {
    "libx264": "medium",
    "libx265": "medium",
    "libsvtav1": "6",
}


class EncodeError(RuntimeError):
    """ffmpeg did not produce a bitstream or a rate measurement."""


@dataclass(frozen=True)
class RatePoint:
    """One CRF encode of one clip."""

    codec: str
    crf: int
    preset: str
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
    bitstream: Path


def encoder_args(codec: str, crf: int, preset: str) -> list[str]:
    """ffmpeg arguments that select the encoder, the rate, and one thread."""
    if codec == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-threads",
            "1",
            "-x264-params",
            "sliced-threads=0",
        ]
    if codec == "libx265":
        return [
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-x265-params",
            "pools=1:frame-threads=1",
        ]
    if codec == "libsvtav1":
        return [
            "-c:v",
            "libsvtav1",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-svtav1-params",
            "lp=1:tune=0",
        ]
    raise ValueError(f"unsupported codec {codec}")


def ffprobe_binary() -> str:
    """ffprobe beside the configured ffmpeg, else the first one on PATH."""
    sibling = Path(ffmpeg_binary()).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    from shutil import which

    found = which("ffprobe")
    if found is None:
        raise EncodeError("ffprobe is not on PATH")
    return found


def video_bits(bitstream: Path) -> int:
    """Sum of video-packet sizes, in bits. Container bytes are not included."""
    completed = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=size",
            "-of",
            "csv=p=0",
            str(bitstream),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "ffprobe failed"
        raise EncodeError(detail)
    total_bytes = 0
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        total_bytes += int(text.split(",")[0])
    if total_bytes < 1:
        raise EncodeError(f"{bitstream} has no video packets")
    return total_bytes * 8


def encode_y4m(
    source: Path,
    bitstream: Path,
    *,
    codec: str,
    crf: int,
    preset: str | None = None,
    measure_decode: bool = True,
) -> RatePoint:
    """Encode ``source`` and, by default, time a single-threaded decode to null.

    Decode time is the decoder only (``-f null``). It does not include
    writing a reconstructed y4m. Callers that need pixels use
    :func:`decode_y4m`. Pass ``measure_decode=False`` when that second
    decode would only repeat this one.
    """
    chosen = BENCHMARK_PRESET[codec] if preset is None else preset
    with source.open("rb") as handle:
        header = parse_header(handle.read(4096))
    # Frame count comes from the payload, not the header. parse_header only
    # sees the first frame tag, so count frames from the file size below.
    info_bytes = source.stat().st_size
    payload = info_bytes - header.header_length
    if payload % header.frame_bytes != 0:
        raise EncodeError(f"{source} is not an integer number of y4m frames")
    frames = payload // header.frame_bytes
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    encode_command = [
        ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        *encoder_args(codec, crf, chosen),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(bitstream),
    ]
    encode_seconds = _run_timed(encode_command, bitstream)
    bits = video_bits(bitstream)
    decode_seconds = 0.0
    if measure_decode:
        decode_command = [
            ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(bitstream),
            "-f",
            "null",
            "-",
        ]
        decode_seconds = _run_timed(decode_command, bitstream)
    return RatePoint(
        codec=codec,
        crf=crf,
        preset=chosen,
        width=header.width,
        height=header.height,
        frames=frames,
        fps_num=header.fps_num,
        fps_den=header.fps_den,
        bits=bits,
        bpp=bits_per_pixel(bits, header.width, header.height, frames),
        kbps=kilobits_per_second(bits, frames, header.fps_num, header.fps_den),
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        bitstream=bitstream,
    )


def decode_y4m(bitstream: Path, dest: Path) -> float:
    """Decode ``bitstream`` to an 8-bit 4:2:0 y4m. Returns wall-clock seconds.

    A failed ffmpeg removes ``dest``. The bitstream is left in place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-threads",
        "1",
        "-i",
        str(bitstream),
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "yuv4mpegpipe",
        str(dest),
    ]
    return _run_timed(command, dest)


def _run_timed(command: list[str], bitstream: Path) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        bitstream.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "ffmpeg failed"
        raise EncodeError(detail)
    return elapsed
