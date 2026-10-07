"""Optional LPIPS.

LPIPS (Zhang et al., 2018) is a distance in a network's feature space.
It is an extra signal, not part of the required CPU baseline: the default
environment does not install PyTorch. If ``torch`` or ``lpips`` cannot be
imported, :func:`lpips_distance` raises :class:`MetricUnavailable` and
:func:`maybe_lpips` returns ``None``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import numpy.typing as npt


class MetricUnavailable(RuntimeError):
    """An optional metric's dependency is not installed."""


def _load_lpips() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        lpips = importlib.import_module("lpips")
    except ImportError as exc:
        raise MetricUnavailable("LPIPS needs the torch and lpips packages") from exc
    return torch, lpips


def lpips_distance(
    reference_rgb: npt.NDArray[np.generic],
    distorted_rgb: npt.NDArray[np.generic],
) -> float:
    """Mean AlexNet LPIPS over frames.

    Arrays are RGB in 0..255, shape ``(height, width, 3)`` or
    ``(frames, height, width, 3)``.
    """
    if reference_rgb.shape != distorted_rgb.shape:
        raise ValueError(f"shape {reference_rgb.shape} != {distorted_rgb.shape}")
    torch, lpips = _load_lpips()
    ref = _as_batch(reference_rgb)
    dist = _as_batch(distorted_rgb)
    loss = lpips.LPIPS(net="alex")
    # LPIPS expects NCHW in [-1, 1].
    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    dist_t = torch.from_numpy(dist).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    with torch.no_grad():
        values = loss(dist_t, ref_t)
    return float(values.mean().item())


def maybe_lpips(
    reference_rgb: npt.NDArray[np.generic],
    distorted_rgb: npt.NDArray[np.generic],
) -> float | None:
    """LPIPS, or ``None`` when torch or lpips is not installed."""
    try:
        return lpips_distance(reference_rgb, distorted_rgb)
    except MetricUnavailable:
        return None


def _as_batch(rgb: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    if rgb.ndim == 3:
        return rgb.astype(np.float64)[None, ...]
    if rgb.ndim == 4:
        return rgb.astype(np.float64)
    raise ValueError(f"RGB must be HWC or THWC, got {rgb.shape}")
