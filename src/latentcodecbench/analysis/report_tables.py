"""Numeric sections of the technical report, rebuilt from the CSVs.

Every mean is a Student-t interval across clips. The prose around these
tables is ``docs/REPORT.md``. A test checks that the report still contains
this text.
"""

from __future__ import annotations

from statistics import fmean

from latentcodecbench.analysis.intervals import (
    MeanInterval,
    classic_bd_rate_samples,
    student_t_interval,
    vae_bd_rate_samples,
)
from latentcodecbench.analysis.propagation import PropagationRow
from latentcodecbench.analysis.propagation_summary import degraded_source_mismatches
from latentcodecbench.analysis.table import ResultRow
from latentcodecbench.analysis.vae_rd import DISPLAY, METRIC_LABEL, VAE_CODECS
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

_CODEC_LABEL = {"libx264": "x264", "libx265": "x265", "libsvtav1": "SVT-AV1"}


def recon_interval_markdown(rows: list[ResultRow]) -> str:
    """Mean reconstruction quality with a 95% Student-t interval across clips."""
    lines = [
        "| VAE | Clips | Mean PSNR-Y | 95% CI | Mean VMAF | 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    grouped: dict[str, list[ResultRow]] = {name: [] for name in SMOKE_ORDER}
    for row in rows:
        grouped.setdefault(row.codec, []).append(row)
    for name in SMOKE_ORDER:
        group = grouped.get(name, [])
        if not group:
            continue
        psnr = student_t_interval([row.psnr_y for row in group])
        vmaf = student_t_interval([row.vmaf for row in group])
        lines.append(
            f"| {name} | {psnr.n} | {psnr.mean:.2f} dB | {_span(psnr, suffix=' dB')} "
            f"| {vmaf.mean:.2f} | {_span(vmaf)} |"
        )
    return "\n".join(lines)


def bdrate_interval_markdown(classic: list[ResultRow], bitrate: list[ResultRow]) -> str:
    """Mean BD-rate against x264 with a 95% Student-t interval across clips in the mean."""
    lines = [
        "| Codec | Metric | Clips in the mean | Mean BD-rate | 95% CI |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for metric in ("psnr_y", "vmaf"):
        samples = classic_bd_rate_samples(classic, metric=metric)
        for codec, values in samples.items():
            if not values:
                continue
            interval = student_t_interval(values)
            span = _span(interval, signed=True, suffix="%")
            lines.append(
                f"| {_CODEC_LABEL.get(codec, codec)} | {METRIC_LABEL[metric]} "
                f"| {interval.n} | {interval.mean:+.2f}% | {span} |"
            )
    vae = vae_bd_rate_samples(classic + bitrate)
    for codec in VAE_CODECS:
        for metric in ("psnr_y", "vmaf"):
            values = vae.get((codec, metric), [])
            if not values:
                continue
            interval = student_t_interval(values)
            span = _span(interval, signed=True, suffix="%")
            lines.append(
                f"| {DISPLAY.get(codec, codec)} | {METRIC_LABEL[metric]} "
                f"| {interval.n} | {interval.mean:+.2f}% | {span} |"
            )
    return "\n".join(lines)


def mismatch_markdown(rows: list[PropagationRow]) -> str:
    """How far ``psnr_degraded_clean`` disagrees across VAEs, from the propagation CSV."""
    found = degraded_source_mismatches(rows)
    pieces: list[str] = []
    for codec, (clips, worst) in found.items():
        label = _CODEC_LABEL.get(codec, codec)
        if clips == 0:
            pieces.append(f"{label} matches on every clip")
        else:
            pieces.append(f"{label} differs on {clips} clips by up to {worst:.3f} dB")
    return "Degraded-source PSNR across VAEs: " + "; ".join(pieces) + "."


def report_numbers(
    classic: list[ResultRow],
    recon: list[ResultRow],
    bitrate: list[ResultRow],
    propagation: list[PropagationRow],
) -> str:
    """The measured sections of ``docs/REPORT.md``."""
    clips = len({row.clip_id for row in recon})
    return "\n\n".join(
        [
            f"Reconstruction means use {clips} clips per VAE. "
            "The interval is a 95% Student-t interval for that mean.",
            recon_interval_markdown(recon),
            "BD-rate intervals use only the clips whose rate is defined. The anchor is x264.",
            bdrate_interval_markdown(classic, bitrate),
            mismatch_markdown(propagation),
        ]
    )


def interval_matches_mean(values: list[float]) -> bool:
    """The reported mean is the arithmetic mean of the same samples."""
    if len(values) < 2:
        return False
    return student_t_interval(values).mean == fmean(values)


def _span(interval: MeanInterval, *, signed: bool = False, suffix: str = "") -> str:
    if signed:
        return f"[{interval.low:+.2f}{suffix}, {interval.high:+.2f}{suffix}]"
    return f"[{interval.low:.2f}{suffix}, {interval.high:.2f}{suffix}]"
