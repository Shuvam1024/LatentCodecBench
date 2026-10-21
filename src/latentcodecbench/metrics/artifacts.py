"""How much source-codec damage is still in a VAE reconstruction.

Three measurements, all on arrays, none of them a VAE forward:

* Latent distance between the posterior mean of the clean source and the
  posterior mean of a degraded source. MSE, relative L2, and cosine.
* Luma PSNR of the degraded-source reconstruction against the clean
  source, and against the degraded source itself. The third PSNR is the
  degraded source against the clean source, so the two reconstruction
  scores have a reference for how damaged the input was.
* Blocking and ringing on luma, and a retention ratio. Retention is 0
  when the reconstruction is as clean as the source, and 1 when it is as
  damaged as the degraded input. A gap smaller than ``1e-6`` has no
  ratio, because the degraded source did not add that artifact.

Blocking is the mean absolute jump across 8x8 boundaries minus the mean
absolute jump in the interior of those blocks, horizontal and vertical
pooled. Ringing is the mean absolute second difference on the samples
immediately beside a strong edge. The edge samples themselves are left
out, so a pure step scores 0 and a one-pixel overshoot beside that step
scores higher. ADR-0012.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from latentcodecbench.metrics.psnr import psnr

_BLOCK = 8
_RINGING_THRESHOLD = 12.0
_RETENTION_EPS = 1e-6


@dataclass(frozen=True)
class LatentDistance:
    """Distance from a clean posterior mean to a degraded one."""

    mse: float
    relative_l2: float
    cosine: float


@dataclass(frozen=True)
class PropagationQuality:
    """Pixel scores for one degraded source through one VAE."""

    psnr_recon_clean: float
    psnr_recon_degraded: float
    psnr_degraded_clean: float
    blocking_clean: float
    blocking_degraded: float
    blocking_recon: float
    blocking_retention: float | None
    ringing_clean: float
    ringing_degraded: float
    ringing_recon: float
    ringing_retention: float | None


def latent_distance(
    clean: npt.NDArray[np.generic],
    other: npt.NDArray[np.generic],
) -> LatentDistance:
    """MSE, ``||clean - other|| / ||clean||``, and cosine similarity.

    A clean tensor with norm 0 uses ``||clean - other||`` as the relative
    L2, since there is no clean length to divide by. Two zero tensors have
    cosine 1. A zero tensor against anything else has cosine 0.
    """
    if clean.shape != other.shape:
        raise ValueError(f"shape {clean.shape} != {other.shape}")
    if clean.size == 0:
        raise ValueError("latent distance needs at least one sample")
    clean_f = clean.astype(np.float64, copy=False)
    other_f = other.astype(np.float64, copy=False)
    diff = clean_f - other_f
    mse = float(np.mean(diff * diff))
    diff_norm = float(np.linalg.norm(diff))
    clean_norm = float(np.linalg.norm(clean_f))
    relative = diff_norm if clean_norm == 0.0 else diff_norm / clean_norm
    other_norm = float(np.linalg.norm(other_f))
    denom = clean_norm * other_norm
    if denom == 0.0:
        cosine = 1.0 if diff_norm == 0.0 else 0.0
    else:
        cosine = float(np.dot(clean_f.ravel(), other_f.ravel()) / denom)
    return LatentDistance(mse=mse, relative_l2=relative, cosine=cosine)


def blocking_score(luma: npt.NDArray[np.generic]) -> float:
    """Boundary jump minus interior jump, on an 8x8 grid.

    ``luma`` is ``(height, width)`` or ``(frames, height, width)``, in
    0-255. Both sides must be divisible by 8 and at least 16. A constant
    field and a one-code ramp score 0. An 8x8 block edge scores high.
    """
    video = _as_frames(luma)
    _frames, height, width = video.shape
    if height % _BLOCK or width % _BLOCK:
        raise ValueError(f"blocking score needs multiples of {_BLOCK}, got {height}x{width}")
    if height < _BLOCK * 2 or width < _BLOCK * 2:
        raise ValueError(f"blocking score needs at least two blocks, got {height}x{width}")
    horizontal = np.abs(video[:, :, 1:] - video[:, :, :-1])
    vertical = np.abs(video[:, 1:, :] - video[:, :-1, :])
    h_boundary = (np.arange(1, width) % _BLOCK) == 0
    v_boundary = (np.arange(1, height) % _BLOCK) == 0
    boundary = np.concatenate(
        [horizontal[:, :, h_boundary].ravel(), vertical[:, v_boundary, :].ravel()]
    )
    interior = np.concatenate(
        [horizontal[:, :, ~h_boundary].ravel(), vertical[:, ~v_boundary, :].ravel()]
    )
    return float(boundary.mean() - interior.mean())


def ringing_score(luma: npt.NDArray[np.generic]) -> float:
    """Mean absolute second difference beside strong edges.

    A strong edge is an absolute neighboring difference of at least 12
    on the 0-255 scale. The two samples that form the edge are excluded.
    The samples one step farther out, on each side that exists, are the
    ones that count. No strong edge scores 0.
    """
    video = _as_frames(luma)
    frames, height, width = video.shape
    rows = video.reshape(frames * height, width)
    cols = np.transpose(video, (0, 2, 1)).reshape(frames * width, height)
    samples = np.concatenate([_beside_abs_second(rows), _beside_abs_second(cols)])
    if samples.size == 0:
        return 0.0
    return float(samples.mean())


def artifact_retention(clean: float, degraded: float, recon: float) -> float | None:
    """``(recon - clean) / (degraded - clean)``.

    0 means the reconstruction dropped the artifact back to the clean
    level. 1 means it kept the degraded level. Above 1 means it is worse
    than the degraded input. ``None`` when the degraded score is within
    ``1e-6`` of the clean score.
    """
    gap = degraded - clean
    if abs(gap) < _RETENTION_EPS:
        return None
    return (recon - clean) / gap


def propagation_quality(
    clean: npt.NDArray[np.generic],
    degraded: npt.NDArray[np.generic],
    recon: npt.NDArray[np.generic],
) -> PropagationQuality:
    """PSNR pair plus blocking and ringing retention. Luma only."""
    if clean.shape != degraded.shape or clean.shape != recon.shape:
        raise ValueError(f"luma shapes {clean.shape}, {degraded.shape}, {recon.shape} must match")
    if clean.ndim != 3:
        raise ValueError(f"luma must be (frames, height, width), got {clean.shape}")
    block_clean = blocking_score(clean)
    block_degraded = blocking_score(degraded)
    block_recon = blocking_score(recon)
    ring_clean = ringing_score(clean)
    ring_degraded = ringing_score(degraded)
    ring_recon = ringing_score(recon)
    return PropagationQuality(
        psnr_recon_clean=psnr(clean, recon),
        psnr_recon_degraded=psnr(degraded, recon),
        psnr_degraded_clean=psnr(clean, degraded),
        blocking_clean=block_clean,
        blocking_degraded=block_degraded,
        blocking_recon=block_recon,
        blocking_retention=artifact_retention(block_clean, block_degraded, block_recon),
        ringing_clean=ring_clean,
        ringing_degraded=ring_degraded,
        ringing_recon=ring_recon,
        ringing_retention=artifact_retention(ring_clean, ring_degraded, ring_recon),
    )


def _as_frames(luma: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    if luma.ndim == 2:
        video = luma[None, ...]
    elif luma.ndim == 3:
        video = luma
    else:
        raise ValueError(f"luma must be 2D or 3D, got ndim {luma.ndim}")
    if video.size == 0:
        raise ValueError("luma is empty")
    return video.astype(np.float64, copy=False)


def _beside_abs_second(lines: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Absolute second differences beside strong jumps along ``lines``.

    ``lines`` has shape ``(count, length)``. First difference index ``i``
    is the jump from sample ``i`` to sample ``i + 1``. The second
    difference is taken at ``i - 1`` and ``i + 2``, which sit one sample
    outside that jump.
    """
    _count, length = lines.shape
    if length < 4:
        return np.empty(0, dtype=np.float64)
    first = lines[:, 1:] - lines[:, :-1]
    strong = np.abs(first) >= _RINGING_THRESHOLD
    second = lines[:, 2:] - 2.0 * lines[:, 1:-1] + lines[:, :-2]
    right_second = second[:, 1:]
    right_strong = strong[:, : right_second.shape[1]]
    left_strong = strong[:, 2:]
    left_second = second[:, : left_strong.shape[1]]
    return np.concatenate([np.abs(right_second[right_strong]), np.abs(left_second[left_strong])])
