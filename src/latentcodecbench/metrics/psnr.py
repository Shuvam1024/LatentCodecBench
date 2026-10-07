"""Peak signal-to-noise ratio.

PSNR is ``10 * log10(MAX^2 / MSE)`` with MAX = 255 for 8-bit video.
One MSE is taken over every sample in the clip, then one PSNR is
reported. Averaging per-frame PSNR would hide a single bad frame inside
the logarithm. Identical inputs have MSE 0 and PSNR of positive infinity.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def psnr(
    reference: npt.NDArray[np.generic],
    distorted: npt.NDArray[np.generic],
    *,
    max_value: float = 255.0,
) -> float:
    """PSNR in dB. Arrays must share a shape. ``max_value`` is the peak."""
    if reference.shape != distorted.shape:
        raise ValueError(f"shape {reference.shape} != {distorted.shape}")
    if reference.size == 0:
        raise ValueError("PSNR needs at least one sample")
    difference = reference.astype(np.float64) - distorted.astype(np.float64)
    error = float(np.mean(difference * difference))
    if error == 0.0:
        return math.inf
    return 10.0 * math.log10((max_value * max_value) / error)
