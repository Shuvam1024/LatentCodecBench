"""BD-rate of the four VAEs against the classic codecs.

The integral is :func:`latentcodecbench.analysis.bdrate.bd_rate`, the same
PCHIP used for x265 and SVT-AV1. This module only decides which samples
enter that integral and which clips stay undefined.

Depth 2 is left out. It sits near 17 dB, under the classic ladder, so it
widens the union and bends the first cubic without adding a shared
operating point. A clip whose remaining points are not strictly
increasing in quality as rate increases is undefined. A clip with no
shared quality range is undefined. The integral is never extended past
the samples. ADR-0011.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Literal

from latentcodecbench.analysis.bdrate import (
    OVERLAP_FLAG,
    BjontegaardError,
    bd_rate,
    overlap_fraction,
)
from latentcodecbench.analysis.table import ResultRow
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

VAE_CODECS: tuple[str, ...] = SMOKE_ORDER
ANCHORS: tuple[str, ...] = ("libx264", "libx265", "libsvtav1")
METRICS: tuple[str, ...] = ("psnr_y", "vmaf")
EXCLUDED_BIT_DEPTH = 2

Status = Literal["defined", "no_overlap", "nonmonotonic", "too_few_points"]

DISPLAY: dict[str, str] = {
    "libx264": "x264",
    "libx265": "x265",
    "libsvtav1": "SVT-AV1",
    "wan2_1": "wan2_1",
    "hunyuan": "hunyuan",
    "ltx": "ltx",
    "wan2_2": "wan2_2",
}
METRIC_LABEL: dict[str, str] = {"psnr_y": "PSNR-Y", "vmaf": "VMAF"}

BD_RATE_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "codec",
    "anchor",
    "metric",
    "status",
    "bd_rate",
    "overlap",
    "below_0_75",
)

_CURVE_ORDER: tuple[str, ...] = ANCHORS + VAE_CODECS
_COLORS: dict[str, str] = {
    "libx264": "#222222",
    "libx265": "#4c78a8",
    "libsvtav1": "#f58518",
    "wan2_1": "#54a24b",
    "hunyuan": "#e45756",
    "ltx": "#b279a2",
    "wan2_2": "#72b7b2",
}


@dataclass(frozen=True)
class ClipBdRate:
    """One VAE against one anchor, on one clip and one metric."""

    clip_id: str
    codec: str
    anchor: str
    metric: str
    status: Status
    bd_rate: float | None
    overlap: float | None
    below_0_75: bool | None


@dataclass(frozen=True)
class BdRateMean:
    """Mean of the defined per-clip BD-rates. Undefined clips are counted."""

    codec: str
    anchor: str
    metric: str
    mean_bd_rate: float | None
    defined: int
    no_overlap: int
    nonmonotonic: int
    too_few_points: int
    mean_overlap: float | None
    below_overlap: int


def vae_clip_bd_rates(rows: list[ResultRow]) -> list[ClipBdRate]:
    """Per-clip BD-rate of each VAE against each classic anchor.

    Depth 2 is omitted from the VAE curve. The anchor keeps every CRF.
    """
    grouped: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
    for row in rows:
        if row.bpp is None or row.crf is None:
            continue
        grouped[(row.clip_id, row.codec)].append(row)
    clips = sorted({clip for clip, codec in grouped if codec in VAE_CODECS})
    found: list[ClipBdRate] = []
    for clip_id in clips:
        for codec in VAE_CODECS:
            for anchor in ANCHORS:
                for metric in METRICS:
                    found.append(
                        _clip_rate(
                            clip_id,
                            codec,
                            anchor,
                            metric,
                            grouped.get((clip_id, codec), []),
                            grouped.get((clip_id, anchor), []),
                        )
                    )
    return found


def vae_bd_rate_means(clips: list[ClipBdRate]) -> list[BdRateMean]:
    """Mean BD-rate of the defined clips, in VAE, anchor, and metric order."""
    grouped: dict[tuple[str, str, str], list[ClipBdRate]] = defaultdict(list)
    for row in clips:
        grouped[(row.codec, row.anchor, row.metric)].append(row)
    means: list[BdRateMean] = []
    for codec in VAE_CODECS:
        for anchor in ANCHORS:
            for metric in METRICS:
                group = grouped.get((codec, anchor, metric), [])
                defined = [
                    row for row in group if row.status == "defined" and row.bd_rate is not None
                ]
                overlaps = [row.overlap for row in defined if row.overlap is not None]
                rates = [row.bd_rate for row in defined if row.bd_rate is not None]
                means.append(
                    BdRateMean(
                        codec=codec,
                        anchor=anchor,
                        metric=metric,
                        mean_bd_rate=fmean(rates) if rates else None,
                        defined=len(defined),
                        no_overlap=sum(row.status == "no_overlap" for row in group),
                        nonmonotonic=sum(row.status == "nonmonotonic" for row in group),
                        too_few_points=sum(row.status == "too_few_points" for row in group),
                        mean_overlap=fmean(overlaps) if overlaps else None,
                        below_overlap=sum(row.below_0_75 is True for row in defined),
                    )
                )
    return means


def vae_bd_rate_markdown(rows: list[ResultRow]) -> str:
    """Markdown table of :func:`vae_bd_rate_means` for the learn note."""
    header = (
        "| VAE | Anchor | Metric | BD-rate | Defined | No overlap | "
        "Non-monotonic | Too few | Mean overlap | Under 75% |"
    )
    lines = [
        header,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in vae_bd_rate_means(vae_clip_bd_rates(rows)):
        if point.mean_bd_rate is None or point.mean_overlap is None:
            rate = "undefined"
            overlap = "undefined"
        else:
            rate = f"{point.mean_bd_rate:+.2f}%"
            overlap = f"{point.mean_overlap * 100:.2f}%"
        anchor = DISPLAY.get(point.anchor, point.anchor)
        metric = METRIC_LABEL.get(point.metric, point.metric)
        lines.append(
            f"| {point.codec} | {anchor} | {metric} | {rate} | {point.defined} | "
            f"{point.no_overlap} | {point.nonmonotonic} | {point.too_few_points} | "
            f"{overlap} | {point.below_overlap} |"
        )
    return "\n".join(lines)


def render_bdrate_csv(clips: list[ClipBdRate]) -> str:
    """CSV text of the per-clip rows, sorted for a stable diff."""
    ordered = sorted(clips, key=lambda row: (row.clip_id, row.codec, row.anchor, row.metric))
    lines = [",".join(BD_RATE_COLUMNS)]
    for row in ordered:
        overlap = "" if row.overlap is None else f"{row.overlap:.6f}"
        rate = "" if row.bd_rate is None else f"{row.bd_rate:.6f}"
        flag = "" if row.below_0_75 is None else ("1" if row.below_0_75 else "0")
        lines.append(
            ",".join(
                (
                    row.clip_id,
                    row.codec,
                    row.anchor,
                    row.metric,
                    row.status,
                    rate,
                    overlap,
                    flag,
                )
            )
        )
    return "\n".join(lines) + "\n"


def write_bdrate_csv(path: Path, clips: list[ClipBdRate]) -> None:
    """Write :func:`render_bdrate_csv` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_bdrate_csv(clips), encoding="utf-8")


def plot_vae_rd(rows: list[ResultRow], out_dir: Path) -> tuple[Path, Path, Path]:
    """Write the mean RD curves and the per-clip BD-rate chart.

    The mean curves keep depth 2, so the floor of the ladder is visible.
    The per-clip chart is PSNR-Y against x264 and omits undefined clips.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    psnr_path = out_dir / "rd-psnr.png"
    vmaf_path = out_dir / "rd-vmaf.png"
    clips_path = out_dir / "rd-bdrate-clips.png"
    _draw_mean(rows, psnr_path, metric="psnr_y")
    _draw_mean(rows, vmaf_path, metric="vmaf")
    _draw_clips(vae_clip_bd_rates(rows), clips_path)
    return psnr_path, vmaf_path, clips_path


def _clip_rate(
    clip_id: str,
    codec: str,
    anchor: str,
    metric: str,
    test_rows: list[ResultRow],
    anchor_rows: list[ResultRow],
) -> ClipBdRate:
    test = _points(test_rows, metric, drop_depth_2=True)
    anchor_points = _points(anchor_rows, metric, drop_depth_2=False)
    overlap = _overlap(anchor_points, test)
    test_status = _curve_status(test)
    anchor_status = _curve_status(anchor_points)
    status: Status
    if test_status == "too_few_points" or anchor_status == "too_few_points":
        status = "too_few_points"
    elif test_status == "nonmonotonic" or anchor_status == "nonmonotonic":
        status = "nonmonotonic"
    elif overlap == 0.0:
        status = "no_overlap"
    else:
        try:
            rate = bd_rate(
                [point[0] for point in anchor_points],
                [point[1] for point in anchor_points],
                [point[0] for point in test],
                [point[1] for point in test],
            )
        except BjontegaardError as exc:
            if "not strictly monotonic" in str(exc):
                status = "nonmonotonic"
                rate = None
            else:
                status = "no_overlap"
                rate = None
        else:
            status = "defined"
        return ClipBdRate(
            clip_id=clip_id,
            codec=codec,
            anchor=anchor,
            metric=metric,
            status=status,
            bd_rate=rate if status == "defined" else None,
            overlap=overlap,
            below_0_75=None if status != "defined" or overlap is None else overlap < OVERLAP_FLAG,
        )
    return ClipBdRate(
        clip_id=clip_id,
        codec=codec,
        anchor=anchor,
        metric=metric,
        status=status,
        bd_rate=None,
        overlap=overlap,
        below_0_75=None,
    )


def _points(
    rows: list[ResultRow],
    metric: str,
    *,
    drop_depth_2: bool,
) -> list[tuple[float, float]]:
    chosen: list[tuple[float, float]] = []
    for row in rows:
        if row.bpp is None:
            continue
        if drop_depth_2 and row.crf == EXCLUDED_BIT_DEPTH:
            continue
        chosen.append((row.bpp, float(getattr(row, metric))))
    chosen.sort(key=lambda point: (point[0], point[1]))
    return chosen


def _curve_status(points: list[tuple[float, float]]) -> Status | None:
    if len(points) < 4:
        return "too_few_points"
    rates = [point[0] for point in points]
    quality = [point[1] for point in points]
    rates_rise = all(later > earlier for earlier, later in pairwise(rates))
    quality_rises = all(later > earlier for earlier, later in pairwise(quality))
    if not rates_rise or not quality_rises:
        return "nonmonotonic"
    return None


def _overlap(
    anchor: list[tuple[float, float]],
    test: list[tuple[float, float]],
) -> float | None:
    if not anchor or not test:
        return None
    return overlap_fraction(
        [point[1] for point in anchor],
        [point[1] for point in test],
    )


def _draw_mean(rows: list[ResultRow], path: Path, *, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, int], list[ResultRow]] = defaultdict(list)
    for row in rows:
        if row.crf is None or row.bpp is None or row.codec not in _CURVE_ORDER:
            continue
        grouped[(row.codec, row.crf)].append(row)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for codec in _CURVE_ORDER:
        points = [(crf, samples) for (name, crf), samples in grouped.items() if name == codec]
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        bpp = [_mean_bpp(samples) for _, samples in points]
        score = [
            sum(float(getattr(row, metric)) for row in samples) / len(samples)
            for _, samples in points
        ]
        axis.plot(
            bpp,
            score,
            marker="o",
            color=_COLORS[codec],
            label=DISPLAY.get(codec, codec),
        )
    axis.set_xscale("log")
    axis.set_xlabel("bits per pixel")
    axis.set_ylabel(METRIC_LABEL[metric])
    axis.set_title(f"Mean {METRIC_LABEL[metric]} across the clip set")
    axis.grid(True, which="both", linestyle=":", linewidth=0.6)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _draw_clips(clips: list[ClipBdRate], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = [
        row
        for row in clips
        if row.anchor == "libx264" and row.metric == "psnr_y" and row.status == "defined"
    ]
    clip_ids = sorted(
        {row.clip_id for row in clips if row.anchor == "libx264" and row.metric == "psnr_y"}
    )
    figure, axis = plt.subplots(figsize=(11.5, 4.6))
    index = {clip_id: position for position, clip_id in enumerate(clip_ids)}
    for codec in VAE_CODECS:
        series = [row for row in wanted if row.codec == codec and row.bd_rate is not None]
        axis.scatter(
            [index[row.clip_id] for row in series],
            [float(row.bd_rate) for row in series if row.bd_rate is not None],
            s=18,
            color=_COLORS[codec],
            label=DISPLAY.get(codec, codec),
        )
    axis.set_xticks(range(len(clip_ids)), clip_ids, rotation=90, fontsize=7)
    axis.set_ylabel("BD-rate vs x264, PSNR-Y (%)")
    axis.set_title("Per-clip BD-rate. Depth 2 is excluded. A missing dot is undefined.")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.6)
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _mean_bpp(samples: list[ResultRow]) -> float:
    values = [row.bpp for row in samples if row.bpp is not None]
    return sum(values) / len(values)
