"""Rate-distortion curves, BD-rate, and the codec-leak study.

Phase 1 stores one CSV of classic-codec operating points and draws the
rate-distortion curves. BD-rate is a PCHIP integral, the piecewise cubic
used by the JCT-VC common test conditions. Phase 2 puts quantized VAE
latents on the same axes. Phase 3 measures how far a re-encoded source
moves the latent.
"""

from latentcodecbench.analysis.bdrate import (
    OVERLAP_FLAG,
    BjontegaardError,
    bd_psnr,
    bd_rate,
    overlap_fraction,
)
from latentcodecbench.analysis.plot import (
    BdRateSummary,
    ClipOverlap,
    mean_bd_rate,
    plot_rd,
    write_overlap,
)
from latentcodecbench.analysis.table import ResultRow, read_results, write_results

__all__ = [
    "OVERLAP_FLAG",
    "BdRateSummary",
    "BjontegaardError",
    "ClipOverlap",
    "ResultRow",
    "bd_psnr",
    "bd_rate",
    "mean_bd_rate",
    "overlap_fraction",
    "plot_rd",
    "read_results",
    "write_overlap",
    "write_results",
]
