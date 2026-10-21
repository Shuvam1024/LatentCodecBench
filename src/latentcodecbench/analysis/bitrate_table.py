"""Mean quality and median rate of the latent-bitrate CSV.

One row of the summary is one VAE at one bit depth: the mean PSNR-Y,
the mean VMAF, and the median bits per pixel across clips. With an even
number of clips the median is the average of the two central values.
``scripts/summarize-vae-bitrate.py`` prints the markdown table.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median

from latentcodecbench.analysis.table import ResultRow
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER
from latentcodecbench.vaes.quantize import BIT_DEPTHS


@dataclass(frozen=True)
class DepthSummary:
    """Mean distortion and median rate at one bit depth."""

    codec: str
    depth: int
    mean_psnr_y: float
    mean_vmaf: float
    median_bpp: float


def bitrate_summaries(rows: list[ResultRow]) -> list[DepthSummary]:
    """One summary per VAE and bit depth, in sweep order."""
    grouped: dict[tuple[str, int], list[ResultRow]] = defaultdict(list)
    for row in rows:
        if row.crf is None:
            raise ValueError(f"{row.clip_id} {row.codec} is missing a bit depth")
        grouped[(row.codec, row.crf)].append(row)
    summaries: list[DepthSummary] = []
    for name in SMOKE_ORDER:
        for depth in BIT_DEPTHS:
            group = grouped.get((name, depth), [])
            if not group:
                raise ValueError(f"no rows for {name} depth {depth}")
            bpps: list[float] = []
            for row in group:
                if row.bpp is None:
                    raise ValueError(f"{row.clip_id} {name} depth {depth} is missing bpp")
                bpps.append(row.bpp)
            summaries.append(
                DepthSummary(
                    codec=name,
                    depth=depth,
                    mean_psnr_y=fmean(row.psnr_y for row in group),
                    mean_vmaf=fmean(row.vmaf for row in group),
                    median_bpp=float(median(bpps)),
                )
            )
    return summaries


def bitrate_summary_markdown(rows: list[ResultRow]) -> str:
    """Markdown table of :func:`bitrate_summaries`.

    PSNR-Y and VMAF are printed to two decimals. Bits per pixel is
    printed to four.
    """
    lines = [
        "| VAE | Depth | Mean PSNR-Y | Mean VMAF | Median bpp |",
        "| --- | --- | --- | --- | --- |",
    ]
    for point in bitrate_summaries(rows):
        lines.append(
            f"| {point.codec} | {point.depth} | {point.mean_psnr_y:.2f} | "
            f"{point.mean_vmaf:.2f} | {point.median_bpp:.4f} |"
        )
    return "\n".join(lines)
