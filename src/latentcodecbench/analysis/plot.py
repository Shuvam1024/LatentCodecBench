"""Rate-distortion curves and the mean BD-rate against an anchor.

Each curve is the mean, across clips, of bits per pixel and of the
metric at that codec's CRF. The x-axis is logarithmic. BD-rate is
computed per clip and then averaged. A clip whose curves do not overlap
is counted and left out of the mean. The overlap fraction of every pair
is kept, including pairs under 75%.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from latentcodecbench.analysis.bdrate import (
    OVERLAP_FLAG,
    BjontegaardError,
    bd_rate,
    overlap_fraction,
)
from latentcodecbench.analysis.table import ResultRow

_DISPLAY = {
    "libx264": "x264",
    "libx265": "x265",
    "libsvtav1": "SVT-AV1",
}
_ORDER = ("libx264", "libx265", "libsvtav1")
_METRICS = {"psnr_y": "PSNR-Y (dB)", "vmaf": "VMAF"}


@dataclass(frozen=True)
class BdRateSummary:
    """Mean BD-rate of one codec against the anchor, on one metric."""

    codec: str
    metric: str
    mean_bd_rate: float
    clips: int
    skipped: int
    mean_overlap: float
    below_overlap: int


@dataclass(frozen=True)
class ClipOverlap:
    """Overlap of one clip, one test codec, one metric, against the anchor."""

    clip_id: str
    anchor: str
    codec: str
    metric: str
    overlap: float


OVERLAP_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "anchor",
    "codec",
    "metric",
    "overlap",
    "below_0_75",
)


def mean_bd_rate(
    rows: list[ResultRow],
    *,
    anchor: str = "libx264",
    metric: str = "psnr_y",
) -> list[BdRateSummary]:
    """Per-codec mean of per-clip BD-rate. The anchor is not in the list."""
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric}")
    by_clip: dict[str, dict[str, list[ResultRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_clip[row.clip_id][row.codec].append(row)
    codecs = [name for name in _ORDER if any(name in codecs for codecs in by_clip.values())]
    summaries: list[BdRateSummary] = []
    for codec in codecs:
        if codec == anchor:
            continue
        values: list[float] = []
        overlaps: list[float] = []
        skipped = 0
        below = 0
        for clip_id in sorted(by_clip):
            anchor_rows = by_clip[clip_id].get(anchor, [])
            test_rows = by_clip[clip_id].get(codec, [])
            if len(anchor_rows) < 4 or len(test_rows) < 4:
                skipped += 1
                continue
            anchor_dist = [float(getattr(row, metric)) for row in anchor_rows]
            test_dist = [float(getattr(row, metric)) for row in test_rows]
            fraction = overlap_fraction(anchor_dist, test_dist)
            overlaps.append(fraction)
            if fraction < OVERLAP_FLAG:
                below += 1
            try:
                values.append(
                    bd_rate(
                        [row.bpp for row in anchor_rows],
                        anchor_dist,
                        [row.bpp for row in test_rows],
                        test_dist,
                    )
                )
            except BjontegaardError:
                skipped += 1
        if not values:
            raise BjontegaardError(f"no overlapping curves for {codec} on {metric}")
        summaries.append(
            BdRateSummary(
                codec=codec,
                metric=metric,
                mean_bd_rate=sum(values) / len(values),
                clips=len(values),
                skipped=skipped,
                mean_overlap=sum(overlaps) / len(overlaps),
                below_overlap=below,
            )
        )
    return summaries


def clip_overlaps(
    rows: list[ResultRow],
    *,
    anchor: str = "libx264",
) -> list[ClipOverlap]:
    """One overlap fraction per clip, test codec, and metric."""
    by_clip: dict[str, dict[str, list[ResultRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_clip[row.clip_id][row.codec].append(row)
    codecs = [name for name in _ORDER if any(name in group for group in by_clip.values())]
    found: list[ClipOverlap] = []
    for clip_id in sorted(by_clip):
        anchor_rows = by_clip[clip_id].get(anchor, [])
        if len(anchor_rows) < 4:
            continue
        for codec in codecs:
            if codec == anchor:
                continue
            test_rows = by_clip[clip_id].get(codec, [])
            if len(test_rows) < 4:
                continue
            for metric in _METRICS:
                found.append(
                    ClipOverlap(
                        clip_id=clip_id,
                        anchor=anchor,
                        codec=codec,
                        metric=metric,
                        overlap=overlap_fraction(
                            [float(getattr(row, metric)) for row in anchor_rows],
                            [float(getattr(row, metric)) for row in test_rows],
                        ),
                    )
                )
    return found


def write_overlap(path: Path, rows: list[ClipOverlap]) -> None:
    """Write per-clip overlap fractions. ``below_0_75`` is 1 when flagged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row.clip_id, row.codec, row.metric))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERLAP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "anchor": row.anchor,
                    "codec": row.codec,
                    "metric": row.metric,
                    "overlap": f"{row.overlap:.6f}",
                    "below_0_75": "1" if row.overlap < OVERLAP_FLAG else "0",
                }
            )


def plot_rd(
    rows: list[ResultRow],
    out_dir: Path,
    *,
    anchor: str = "libx264",
    overlap_out: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``rd-psnr.png`` and ``rd-vmaf.png``. Return the two paths.

    When ``overlap_out`` is set, also write the per-clip overlap fractions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    psnr_path = out_dir / "rd-psnr.png"
    vmaf_path = out_dir / "rd-vmaf.png"
    _draw(rows, psnr_path, metric="psnr_y")
    _draw(rows, vmaf_path, metric="vmaf")
    for summary in mean_bd_rate(rows, anchor=anchor, metric="psnr_y"):
        _print_summary(summary, anchor)
    for summary in mean_bd_rate(rows, anchor=anchor, metric="vmaf"):
        _print_summary(summary, anchor)
    if overlap_out is not None:
        write_overlap(overlap_out, clip_overlaps(rows, anchor=anchor))
    return psnr_path, vmaf_path


def _draw(rows: list[ResultRow], path: Path, *, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, int], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.codec, row.crf)].append(row)
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for codec in _ORDER:
        points = [(crf, samples) for (name, crf), samples in grouped.items() if name == codec]
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        bpp = [sum(row.bpp for row in samples) / len(samples) for _, samples in points]
        score = [
            sum(float(getattr(row, metric)) for row in samples) / len(samples)
            for _, samples in points
        ]
        axis.plot(bpp, score, marker="o", label=_DISPLAY.get(codec, codec))
    axis.set_xscale("log")
    axis.set_xlabel("bits per pixel")
    axis.set_ylabel(_METRICS[metric])
    axis.set_title(f"Mean {_METRICS[metric]} across the clip set")
    axis.grid(True, which="both", linestyle=":", linewidth=0.6)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _print_summary(summary: BdRateSummary, anchor: str) -> None:
    name = _DISPLAY.get(summary.codec, summary.codec)
    anchor_name = _DISPLAY.get(anchor, anchor)
    flag = f", {summary.below_overlap} under 75%" if summary.below_overlap else ""
    print(
        f"BD-rate {name} vs {anchor_name} on {summary.metric}: "
        f"{summary.mean_bd_rate:.2f}% over {summary.clips} clips, "
        f"mean overlap {summary.mean_overlap * 100:.2f}%"
        + (f" ({summary.skipped} skipped)" if summary.skipped else "")
        + flag
    )
