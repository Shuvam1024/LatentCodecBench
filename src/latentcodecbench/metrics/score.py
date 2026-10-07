"""Score one distorted y4m against its canonical reference.

PSNR-Y, SSIM, and flicker use luma. PSNR-RGB converts both clips with
the BT.709 limited-range matrix in :mod:`latentcodecbench.metrics.color`.
VMAF is libvmaf's pooled mean. LPIPS is included only when torch and
lpips import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from latentcodecbench.data.y4m import read_y4m
from latentcodecbench.metrics.color import ycbcr_to_rgb
from latentcodecbench.metrics.flicker import temporal_flicker
from latentcodecbench.metrics.perceptual import maybe_lpips
from latentcodecbench.metrics.psnr import psnr
from latentcodecbench.metrics.ssim import ssim
from latentcodecbench.metrics.vmaf import vmaf


@dataclass(frozen=True)
class ClipScore:
    """One distorted clip against its reference."""

    psnr_y: float
    psnr_rgb: float
    ssim: float
    flicker: float
    vmaf: float
    lpips: float | None


def score_y4m(reference: Path, distorted: Path) -> ClipScore:
    """Score ``distorted`` against ``reference``. Both are y4m paths."""
    _, ref_y, ref_u, ref_v = read_y4m(reference)
    _, dist_y, dist_u, dist_v = read_y4m(distorted)
    if ref_y.shape != dist_y.shape:
        raise ValueError(f"luma shape {ref_y.shape} != {dist_y.shape}")
    ref_rgb = ycbcr_to_rgb(ref_y, ref_u, ref_v)
    dist_rgb = ycbcr_to_rgb(dist_y, dist_u, dist_v)
    return ClipScore(
        psnr_y=psnr(ref_y, dist_y),
        psnr_rgb=psnr(ref_rgb, dist_rgb),
        ssim=ssim(ref_y, dist_y),
        flicker=temporal_flicker(ref_y, dist_y),
        vmaf=vmaf(distorted, reference),
        lpips=maybe_lpips(ref_rgb, dist_rgb),
    )
