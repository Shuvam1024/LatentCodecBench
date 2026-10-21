"""Distortion and temporal metrics.

PSNR (luma and RGB), SSIM on luma, VMAF via ffmpeg libvmaf, and a
temporal-flicker score. LPIPS is optional and returns nothing when torch
is not installed. Optical-flow warping error is documented as future work
in the flicker module: it needs a flow model this CPU baseline does not ship.
Phase 3 adds latent distance and blocking/ringing retention on luma.
"""

from latentcodecbench.metrics.artifacts import (
    LatentDistance,
    PropagationQuality,
    artifact_retention,
    blocking_score,
    latent_distance,
    propagation_quality,
    ringing_score,
)
from latentcodecbench.metrics.flicker import temporal_flicker
from latentcodecbench.metrics.perceptual import MetricUnavailable, lpips_distance, maybe_lpips
from latentcodecbench.metrics.psnr import psnr
from latentcodecbench.metrics.score import ClipScore, score_y4m
from latentcodecbench.metrics.ssim import ssim
from latentcodecbench.metrics.vmaf import vmaf

__all__ = [
    "ClipScore",
    "LatentDistance",
    "MetricUnavailable",
    "PropagationQuality",
    "artifact_retention",
    "blocking_score",
    "latent_distance",
    "lpips_distance",
    "maybe_lpips",
    "propagation_quality",
    "psnr",
    "ringing_score",
    "score_y4m",
    "ssim",
    "temporal_flicker",
    "vmaf",
]
