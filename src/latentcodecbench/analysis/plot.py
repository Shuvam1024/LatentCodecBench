"""Rate-distortion curves and the mean BD-rate against an anchor.

Each curve is the mean, across clips, of bits per pixel and of the
metric at that codec's CRF. The x-axis is logarithmic. BD-rate is
computed per clip and then averaged. A clip whose curves do not overlap
is counted and left out of the mean.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from latentcodecbench.analysis.bdrate import BjontegaardError, bd_rate
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
        skipped = 0
        for clip_id in sorted(by_clip):
            anchor_rows = by_clip[clip_id].get(anchor, [])
            test_rows = by_clip[clip_id].get(codec, [])
            if len(anchor_rows) < 4 or len(test_rows) < 4:
                skipped += 1
                continue
            try:
                values.append(
                    bd_rate(
                        [row.bpp for row in anchor_rows],
                        [float(getattr(row, metric)) for row in anchor_rows],
                        [row.bpp for row in test_rows],
                        [float(getattr(row, metric)) for row in test_rows],
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
            )
        )
    return summaries


def plot_rd(
    rows: list[ResultRow],
    out_dir: Path,
    *,
    anchor: str = "libx264",
) -> tuple[Path, Path]:
    """Write ``rd-psnr.png`` and ``rd-vmaf.png``. Return the two paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    psnr_path = out_dir / "rd-psnr.png"
    vmaf_path = out_dir / "rd-vmaf.png"
    _draw(rows, psnr_path, metric="psnr_y")
    _draw(rows, vmaf_path, metric="vmaf")
    for summary in mean_bd_rate(rows, anchor=anchor, metric="psnr_y"):
        _print_summary(summary, anchor)
    for summary in mean_bd_rate(rows, anchor=anchor, metric="vmaf"):
        _print_summary(summary, anchor)
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
    print(
        f"BD-rate {name} vs {anchor_name} on {summary.metric}: "
        f"{summary.mean_bd_rate:.2f}% over {summary.clips} clips"
        + (f" ({summary.skipped} skipped)" if summary.skipped else "")
    )
