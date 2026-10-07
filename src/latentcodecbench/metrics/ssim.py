"""Structural similarity on luma, Wang et al. 2004.

The window is the paper's 11x11 Gaussian with sigma 1.5, normalized to
sum to 1. Stabilizers are ``k1 = 0.01`` and ``k2 = 0.03`` with ``L = 255``,
so ``C1 = (k1 L)^2`` and ``C2 = (k2 L)^2``. Local statistics use that
normalized window (population form, not the ``N-1`` sample correction).
A clip score is the mean of per-frame mean-SSIM.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_WINDOW = 11
_SIGMA = 1.5
_K1 = 0.01
_K2 = 0.03


def _gaussian_kernel(window: int = _WINDOW, sigma: float = _SIGMA) -> npt.NDArray[np.float64]:
    coords = np.arange(window, dtype=np.float64) - (window // 2)
    kernel = np.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def _filter_valid(
    image: npt.NDArray[np.float64], kernel: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Separable valid convolution with a symmetric 1-D kernel."""
    k = kernel.size
    horizontal = np.lib.stride_tricks.sliding_window_view(image, k, axis=1) @ kernel
    return np.lib.stride_tricks.sliding_window_view(horizontal, k, axis=0) @ kernel


def ssim_plane(
    reference: npt.NDArray[np.generic],
    distorted: npt.NDArray[np.generic],
    *,
    max_value: float = 255.0,
) -> float:
    """Mean SSIM of one luma plane. Both arrays are ``(height, width)``."""
    if reference.shape != distorted.shape:
        raise ValueError(f"shape {reference.shape} != {distorted.shape}")
    if reference.ndim != 2:
        raise ValueError("ssim_plane expects a single plane")
    height, width = reference.shape
    if height < _WINDOW or width < _WINDOW:
        raise ValueError(f"SSIM needs at least {_WINDOW}x{_WINDOW}, got {height}x{width}")
    ref = reference.astype(np.float64)
    dist = distorted.astype(np.float64)
    kernel = _gaussian_kernel()
    mu_ref = _filter_valid(ref, kernel)
    mu_dist = _filter_valid(dist, kernel)
    sigma_ref = _filter_valid(ref * ref, kernel) - mu_ref * mu_ref
    sigma_dist = _filter_valid(dist * dist, kernel) - mu_dist * mu_dist
    sigma_cross = _filter_valid(ref * dist, kernel) - mu_ref * mu_dist
    c1 = (_K1 * max_value) ** 2
    c2 = (_K2 * max_value) ** 2
    numerator = (2.0 * mu_ref * mu_dist + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_ref * mu_ref + mu_dist * mu_dist + c1) * (sigma_ref + sigma_dist + c2)
    return float(np.mean(numerator / denominator))


def ssim(
    reference: npt.NDArray[np.generic],
    distorted: npt.NDArray[np.generic],
    *,
    max_value: float = 255.0,
) -> float:
    """Mean SSIM. Arrays are ``(height, width)`` or ``(frames, height, width)``."""
    if reference.shape != distorted.shape:
        raise ValueError(f"shape {reference.shape} != {distorted.shape}")
    if reference.ndim == 2:
        return ssim_plane(reference, distorted, max_value=max_value)
    if reference.ndim != 3:
        raise ValueError(f"SSIM expects 2-D or 3-D arrays, got {reference.ndim}")
    scores = [
        ssim_plane(reference[frame], distorted[frame], max_value=max_value)
        for frame in range(reference.shape[0])
    ]
    return float(np.mean(scores))
