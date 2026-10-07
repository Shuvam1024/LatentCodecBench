"""Deterministic stand-in clips for tests.

Real videos are large and are not part of the test suite. Anything that
needs a video-shaped array calls :func:`make_synthetic_clip`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def make_synthetic_clip(
    *,
    num_frames: int = 8,
    height: int = 32,
    width: int = 32,
    num_channels: int = 3,
    seed: int = 0,
) -> npt.NDArray[np.uint8]:
    """Return a uint8 clip of shape ``(num_frames, height, width, num_channels)``.

    The same seed always yields the same array. A seeded base image plus a
    square that steps across frames gives later tests both texture and motion.
    """
    for name, value in (
        ("num_frames", num_frames),
        ("height", height),
        ("width", width),
        ("num_channels", num_channels),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")

    rng = np.random.default_rng(seed)
    base: npt.NDArray[np.int32] = rng.integers(
        0,
        64,
        size=(height, width, num_channels),
        dtype=np.int32,
    )
    side = max(1, min(height, width) // 8)
    frames: list[npt.NDArray[np.uint8]] = []
    for t in range(num_frames):
        frame = np.clip(base + t, 0, 255).astype(np.uint8)
        top = (t * 3) % height
        left = (t * 5) % width
        frame[top : min(top + side, height), left : min(left + side, width), :] = np.uint8(255)
        frames.append(frame)
    stacked = np.stack(frames, axis=0)
    return stacked
