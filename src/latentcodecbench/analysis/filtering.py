"""What the propagation rows say about filtering compressed training video.

The pixel number is the per-clip change in luma PSNR of the reconstruction,
scored against the clean source::

    delta = psnr_recon_clean - clean_psnr_y

``clean_psnr_y`` is the Phase 2 reconstruction of the same clip and VAE.
Positive means the compressed source came back closer to the clean pixels
than the clean source did. Median latent relative L2 is reported beside
it: a delta near zero is not a latent that stayed put. ADR-0012.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from latentcodecbench.analysis.propagation import PropagationRow
from latentcodecbench.analysis.propagation_summary import spread
from latentcodecbench.analysis.table import ResultRow
from latentcodecbench.codecs.degrade import STUDY_LADDER
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

_CODEC_LABEL = {"libx264": "x264", "libx265": "x265", "libsvtav1": "SVT-AV1"}
_RUNG_NAME = {0: "mildest", 1: "middle", 2: "harshest"}


@dataclass(frozen=True)
class FilterPoint:
    """One VAE at one degraded-source point, summarized over clips."""

    vae: str
    source_codec: str
    source_crf: int
    clips: int
    delta_median: float
    delta_q1: float
    delta_q3: float
    clips_above_clean: int
    latent_relative_l2_median: float


def filter_points(rows: list[PropagationRow], recon: list[ResultRow]) -> list[FilterPoint]:
    """Median pixel change and median latent shift at each measured point.

    ``recon`` is the clean-source reconstruction table. A propagation row
    with no matching clip and VAE, or two clean rows for one pair, raises
    ``ValueError``.
    """
    clean: dict[tuple[str, str], float] = {}
    for clean_row in recon:
        key = (clean_row.clip_id, clean_row.codec)
        if key in clean:
            raise ValueError(
                f"duplicate clean reconstruction for {clean_row.clip_id} {clean_row.codec}"
            )
        clean[key] = clean_row.psnr_y

    deltas: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    shifts: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for prop in rows:
        key = (prop.clip_id, prop.vae)
        if key not in clean:
            raise ValueError(f"no clean reconstruction for {prop.clip_id} {prop.vae}")
        group = (prop.vae, prop.source_codec, prop.source_crf)
        deltas[group].append(prop.psnr_recon_clean - clean[key])
        shifts[group].append(prop.latent_relative_l2)

    ordered: list[tuple[str, str, int]] = []
    for vae in SMOKE_ORDER:
        for codec, crfs in STUDY_LADDER.items():
            for crf in crfs:
                group = (vae, codec, crf)
                if group in deltas:
                    ordered.append(group)
    for group in sorted(deltas):
        if group not in ordered:
            ordered.append(group)

    points: list[FilterPoint] = []
    for vae, codec, crf in ordered:
        pixel = spread(deltas[(vae, codec, crf)])
        latent = spread(shifts[(vae, codec, crf)])
        points.append(
            FilterPoint(
                vae=vae,
                source_codec=codec,
                source_crf=crf,
                clips=len(deltas[(vae, codec, crf)]),
                delta_median=pixel.median,
                delta_q1=pixel.q1,
                delta_q3=pixel.q3,
                clips_above_clean=sum(value > 0 for value in deltas[(vae, codec, crf)]),
                latent_relative_l2_median=latent.median,
            )
        )
    return points


def rising_latent_curves(rows: list[PropagationRow]) -> tuple[int, int]:
    """How many clip, VAE, and codec curves have relative L2 rising with CRF.

    Returns ``(rising, total)``. A curve rises when each higher CRF has a
    strictly larger relative L2 than the CRF below it.
    """
    grouped: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        grouped[(row.clip_id, row.vae, row.source_codec)][row.source_crf] = row.latent_relative_l2
    rising = 0
    for points in grouped.values():
        ordered = [points[crf] for crf in sorted(points)]
        if len(ordered) >= 2 and all(later > earlier for earlier, later in pairwise(ordered)):
            rising += 1
    return rising, len(grouped)


def filtering_report(rows: list[PropagationRow], recon: list[ResultRow]) -> str:
    """Markdown table and the bounds a filter can cite. Both come from the rows."""
    if not rows:
        raise ValueError("no propagation rows")
    points = filter_points(rows, recon)
    if not points:
        raise ValueError("no filter points")
    clips = len({row.clip_id for row in rows})
    vaes = len({row.vae for row in rows})
    sources = len({(row.source_codec, row.source_crf) for row in rows})
    lines = [
        f"{len(rows)} rows, {clips} clips, {vaes} VAEs, {sources} degraded sources. "
        "Median ΔPSNR is the median over clips of reconstruction PSNR against the "
        "clean source, minus that clip's clean-source reconstruction PSNR. "
        "Positive means the compressed source scored higher.",
        "",
        "| VAE | Source | CRF | Median ΔPSNR | IQR | Clips above clean recon | "
        "Median latent rel. L2 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for point in points:
        lines.append(
            f"| {point.vae} | {_CODEC_LABEL.get(point.source_codec, point.source_codec)} "
            f"| {point.source_crf} | {point.delta_median:+.3f} dB "
            f"| [{point.delta_q1:+.3f}, {point.delta_q3:+.3f}] "
            f"| {point.clips_above_clean} of {point.clips} "
            f"| {point.latent_relative_l2_median:.3f} |"
        )
    lines.append("")
    lines.extend(_rung_lines(points))
    lines.append(_ratio_line(points))
    lines.append("")
    lines.append(_shift_line(points, rows))
    return "\n".join(lines)


def _rung(codec: str, crf: int) -> int:
    return STUDY_LADDER[codec].index(crf)


def _rung_lines(points: list[FilterPoint]) -> list[str]:
    lines: list[str] = []
    for rung, name in _RUNG_NAME.items():
        band = [point for point in points if _rung(point.source_codec, point.source_crf) == rung]
        labels: list[str] = []
        for point in band:
            label = f"{_CODEC_LABEL[point.source_codec]} CRF {point.source_crf}"
            if label not in labels:
                labels.append(label)
        lowest = min(band, key=lambda point: point.delta_median)
        highest = max(band, key=lambda point: point.delta_median)
        fewest = min(band, key=lambda point: point.clips_above_clean)
        most = max(band, key=lambda point: point.clips_above_clean)
        near = min(band, key=lambda point: point.latent_relative_l2_median)
        far = max(band, key=lambda point: point.latent_relative_l2_median)
        lines.append(
            f"The {name} rung is {', '.join(labels)}. "
            f"Across {len(band)} groups the median ΔPSNR runs from "
            f"{lowest.delta_median:+.3f} dB ({_where(lowest)}) to "
            f"{highest.delta_median:+.3f} dB ({_where(highest)}). "
            f"Clips above the clean reconstruction run from "
            f"{fewest.clips_above_clean} of {fewest.clips} to "
            f"{most.clips_above_clean} of {most.clips}. "
            f"Median latent relative L2 runs from {near.latent_relative_l2_median:.3f} "
            f"({_where(near)}) to {far.latent_relative_l2_median:.3f} ({_where(far)})."
        )
        lines.append("")
    return lines


def _ratio_line(points: list[FilterPoint]) -> str:
    pieces: list[str] = []
    for vae in SMOKE_ORDER:
        mild = _point(points, vae, "libx264", 0)
        harsh = _point(points, vae, "libx264", 2)
        if mild is None or harsh is None or mild.latent_relative_l2_median == 0:
            continue
        ratio = harsh.latent_relative_l2_median / mild.latent_relative_l2_median
        pieces.append(
            f"{vae} {ratio:.2f} "
            f"({harsh.latent_relative_l2_median:.3f} / {mild.latent_relative_l2_median:.3f})"
        )
    return (
        "Ratio of the x264 median latent relative L2 at the harshest rung "
        f"to the mildest rung: {', '.join(pieces)}."
    )


def _shift_line(points: list[FilterPoint], rows: list[PropagationRow]) -> str:
    smallest = min(points, key=lambda point: point.latent_relative_l2_median)
    rising, total = rising_latent_curves(rows)
    return (
        f"The smallest median latent relative L2 in the table is "
        f"{smallest.latent_relative_l2_median:.3f} ({_where(smallest)}). "
        f"Relative L2 rises with CRF on {rising} of {total} clip, VAE, and codec curves."
    )


def _point(points: list[FilterPoint], vae: str, codec: str, rung: int) -> FilterPoint | None:
    for point in points:
        if (
            point.vae == vae
            and point.source_codec == codec
            and _rung(codec, point.source_crf) == rung
        ):
            return point
    return None


def _where(point: FilterPoint) -> str:
    codec = _CODEC_LABEL.get(point.source_codec, point.source_codec)
    return f"{point.vae}, {codec} CRF {point.source_crf}"
