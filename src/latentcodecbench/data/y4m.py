"""Read and write the subset of YUV4MPEG2 this benchmark uses.

The canonical clips are 8-bit 4:2:0. The reader also accepts the 4:2:0
header variants that show up in the Xiph derf files (including a missing
``C`` tag, which the spec treats as 4:2:0 JPEG siting).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

_CHROMA_420 = frozenset({"420", "420jpeg", "420mpeg2", "420paldv"})


@dataclass(frozen=True)
class Y4MInfo:
    """Geometry parsed from a YUV4MPEG2 header."""

    width: int
    height: int
    fps_num: int
    fps_den: int
    colorspace: str
    header_length: int
    frame_header_length: int

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def luma_bytes(self) -> int:
        return self.width * self.height

    @property
    def payload_bytes(self) -> int:
        if self.colorspace not in _CHROMA_420:
            raise ValueError(f"only 4:2:0 y4m is supported, got {self.colorspace}")
        return self.width * self.height * 3 // 2

    @property
    def frame_bytes(self) -> int:
        return self.frame_header_length + self.payload_bytes


def parse_header(blob: bytes) -> Y4MInfo:
    """Parse a YUV4MPEG2 header and the first frame header."""
    try:
        newline = blob.index(b"\n")
    except ValueError as exc:
        raise ValueError("y4m header has no newline") from exc
    line = blob[:newline].decode("ascii")
    if not line.startswith("YUV4MPEG2"):
        raise ValueError(f"not a y4m header: {line!r}")
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        fields[token[0]] = token[1:]
    try:
        width = int(fields["W"])
        height = int(fields["H"])
        fps_num_s, fps_den_s = fields["F"].split(":")
        fps_num = int(fps_num_s)
        fps_den = int(fps_den_s)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"incomplete y4m header: {line}") from exc
    if width < 2 or height < 2 or fps_num < 1 or fps_den < 1:
        raise ValueError(f"invalid y4m geometry: {line}")
    colorspace = fields.get("C", "420jpeg")
    header_length = newline + 1
    rest = blob[header_length:]
    if not rest.startswith(b"FRAME"):
        raise ValueError("y4m frame header does not start with FRAME")
    try:
        frame_newline = rest.index(b"\n")
    except ValueError as exc:
        raise ValueError("y4m frame header has no newline") from exc
    return Y4MInfo(
        width=width,
        height=height,
        fps_num=fps_num,
        fps_den=fps_den,
        colorspace=colorspace,
        header_length=header_length,
        frame_header_length=frame_newline + 1,
    )


def read_y4m(
    path: Path,
) -> tuple[Y4MInfo, npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Return ``(info, luma, u, v)``.

    ``luma`` has shape ``(T, H, W)``. ``u`` and ``v`` have shape
    ``(T, H/2, W/2)``.
    """
    blob = path.read_bytes()
    info = parse_header(blob)
    if info.colorspace not in _CHROMA_420:
        raise ValueError(f"{path} uses {info.colorspace}, want 4:2:0")
    if info.width % 2 or info.height % 2:
        raise ValueError(f"{path} is {info.width}x{info.height}; 4:2:0 needs even sides")
    available = len(blob) - info.header_length
    if available % info.frame_bytes != 0:
        raise ValueError(f"{path} is not an integer number of frames")
    num_frames = available // info.frame_bytes
    luma = np.empty((num_frames, info.height, info.width), dtype=np.uint8)
    chroma_h = info.height // 2
    chroma_w = info.width // 2
    u_plane = np.empty((num_frames, chroma_h, chroma_w), dtype=np.uint8)
    v_plane = np.empty((num_frames, chroma_h, chroma_w), dtype=np.uint8)
    offset = info.header_length
    luma_bytes = info.luma_bytes
    chroma_bytes = chroma_h * chroma_w
    for index in range(num_frames):
        if blob[offset : offset + 5] != b"FRAME":
            raise ValueError(f"{path} frame {index} is missing its FRAME tag")
        pixel = offset + info.frame_header_length
        luma[index] = np.frombuffer(blob[pixel : pixel + luma_bytes], dtype=np.uint8).reshape(
            info.height, info.width
        )
        u_start = pixel + luma_bytes
        v_start = u_start + chroma_bytes
        u_bytes = blob[u_start:v_start]
        u_plane[index] = np.frombuffer(u_bytes, dtype=np.uint8).reshape(chroma_h, chroma_w)
        v_plane[index] = np.frombuffer(
            blob[v_start : v_start + chroma_bytes], dtype=np.uint8
        ).reshape(chroma_h, chroma_w)
        offset += info.frame_bytes
    return info, luma, u_plane, v_plane


def write_y4m(
    path: Path,
    luma: npt.NDArray[np.uint8],
    u_plane: npt.NDArray[np.uint8],
    v_plane: npt.NDArray[np.uint8],
    *,
    fps_num: int = 25,
    fps_den: int = 1,
) -> None:
    """Write an 8-bit 4:2:0 y4m file. Planes are ``(T, H, W)`` and ``(T, H/2, W/2)``."""
    if luma.ndim != 3 or luma.dtype != np.uint8:
        raise ValueError("luma must be uint8 with shape (T, H, W)")
    num_frames, height, width = luma.shape
    if u_plane.shape != (num_frames, height // 2, width // 2) or v_plane.shape != u_plane.shape:
        raise ValueError("chroma planes must be (T, H/2, W/2) uint8")
    header = f"YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} Ip A1:1 C420mpeg2\n".encode(
        "ascii"
    )
    frame_tag = b"FRAME\n"
    with path.open("wb") as handle:
        handle.write(header)
        for index in range(num_frames):
            handle.write(frame_tag)
            handle.write(np.ascontiguousarray(luma[index]).tobytes())
            handle.write(np.ascontiguousarray(u_plane[index]).tobytes())
            handle.write(np.ascontiguousarray(v_plane[index]).tobytes())
