"""Summary tables and figures for the artifact-propagation CSV.

One summary row is one VAE, one source codec, and one CRF point, pooled
over the 34 clips. Every number is a median with its interquartile range,
because the per-clip retention ratios are heavy-tailed.

Retention is ``(recon - clean) / (degraded - clean)``. When the codec
barely moved an artifact score, the denominator is close to zero and one
clip can produce a ratio of -1403. The median and IQR ignore those tails
without picking a threshold. The pooled ratio ``sum(recon - clean) /
sum(degraded - clean)`` over the clips is reported next to it: it weights
each clip by how much the codec actually added, so a near-zero gap
contributes almost nothing. ADR-0012.

The clean reconstruction PSNR comes from the Phase 2 sweep
(``docs/results/vae-recon.csv``, ``psnr_y``), on the same clips, VAEs,
and luma PSNR definition. Under independent errors the expected PSNR of a
degraded-source reconstruction is
``-10 log10(10^(-a/10) + 10^(-b/10))`` with ``a`` the clean VAE PSNR and
``b`` the degraded-source PSNR. The measured minus that prediction is the
``additive_gap_db`` column: positive means the VAE hid some of the codec
damage, negative means the two errors compounded.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latentcodecbench.analysis.propagation import PropagationRow
from latentcodecbench.analysis.table import ResultRow
from latentcodecbench.codecs.degrade import study_points
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

_CODEC_LABEL = {"libx264": "x264", "libx265": "x265", "libsvtav1": "SVT-AV1"}
_CODEC_ORDER = ("libx264", "libx265", "libsvtav1")
# Two degraded files with PSNR closer than this are treated as the same file.
SAME_FILE_DB = 1e-9


@dataclass(frozen=True)
class Spread:
    """Median and quartiles of one column over a group of clips."""

    median: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


@dataclass(frozen=True)
class GroupSummary:
    """One VAE at one degraded-source point, over all clips."""

    vae: str
    source_codec: str
    source_crf: int
    clips: int
    psnr_degraded_clean: Spread
    psnr_recon_clean: Spread
    psnr_recon_degraded: Spread
    clean_recon_psnr: float | None
    additive_gap_db: float | None
    latent_cosine: Spread
    latent_relative_l2: Spread
    blocking_retention: Spread
    blocking_pooled: float
    ringing_retention: Spread
    ringing_pooled: float
    ringing_gap: Spread


SUMMARY_COLUMNS: tuple[str, ...] = (
    "vae",
    "source_codec",
    "source_crf",
    "clips",
    "psnr_degraded_clean_median",
    "psnr_recon_clean_median",
    "psnr_recon_degraded_median",
    "clean_recon_psnr_median",
    "additive_gap_db_median",
    "latent_cosine_median",
    "latent_relative_l2_median",
    "latent_relative_l2_q1",
    "latent_relative_l2_q3",
    "blocking_retention_median",
    "blocking_retention_q1",
    "blocking_retention_q3",
    "blocking_retention_pooled",
    "ringing_retention_median",
    "ringing_retention_q1",
    "ringing_retention_q3",
    "ringing_retention_pooled",
    "ringing_gap_median",
)


def spread(values: Iterable[float]) -> Spread:
    """Median, Q1 and Q3 with numpy's linear interpolation. Empty is NaN."""
    data = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if data.size == 0:
        nan = float("nan")
        return Spread(nan, nan, nan)
    q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    return Spread(float(median), float(q1), float(q3))


def pooled_retention(
    clean: Iterable[float], degraded: Iterable[float], recon: Iterable[float]
) -> float:
    """``sum(recon - clean) / sum(degraded - clean)``. NaN when the gap sums to 0."""
    c = np.asarray(list(clean), dtype=float)
    d = np.asarray(list(degraded), dtype=float)
    r = np.asarray(list(recon), dtype=float)
    gap = float(np.sum(d - c))
    if abs(gap) < 1e-12:
        return float("nan")
    return float(np.sum(r - c) / gap)


def additive_psnr(clean_recon_db: float, degraded_db: float) -> float:
    """PSNR of two independent errors with these PSNRs, added in MSE."""
    return -10.0 * math.log10(10.0 ** (-clean_recon_db / 10.0) + 10.0 ** (-degraded_db / 10.0))


def summarize(
    rows: list[PropagationRow],
    clean_recon: list[ResultRow] | None = None,
) -> list[GroupSummary]:
    """One :class:`GroupSummary` per VAE and study point, in study order."""
    baseline: dict[tuple[str, str], float] = {}
    for recon in clean_recon or []:
        baseline[(recon.clip_id, recon.codec)] = recon.psnr_y
    groups: dict[tuple[str, str, int], list[PropagationRow]] = defaultdict(list)
    for row in rows:
        groups[(row.vae, row.source_codec, row.source_crf)].append(row)
    out: list[GroupSummary] = []
    vaes = [v for v in SMOKE_ORDER if any(k[0] == v for k in groups)]
    for vae in vaes:
        for codec, crf in study_points():
            group = groups.get((vae, codec, crf), [])
            if not group:
                continue
            clean_db = [baseline[(r.clip_id, vae)] for r in group if (r.clip_id, vae) in baseline]
            gaps = [
                r.psnr_recon_clean
                - additive_psnr(baseline[(r.clip_id, vae)], r.psnr_degraded_clean)
                for r in group
                if (r.clip_id, vae) in baseline
            ]
            out.append(
                GroupSummary(
                    vae=vae,
                    source_codec=codec,
                    source_crf=crf,
                    clips=len(group),
                    psnr_degraded_clean=spread(r.psnr_degraded_clean for r in group),
                    psnr_recon_clean=spread(r.psnr_recon_clean for r in group),
                    psnr_recon_degraded=spread(r.psnr_recon_degraded for r in group),
                    clean_recon_psnr=spread(clean_db).median if clean_db else None,
                    additive_gap_db=spread(gaps).median if gaps else None,
                    latent_cosine=spread(r.latent_cosine for r in group),
                    latent_relative_l2=spread(r.latent_relative_l2 for r in group),
                    blocking_retention=spread(_values(group, "blocking_retention")),
                    blocking_pooled=pooled_retention(
                        (r.blocking_clean for r in group),
                        (r.blocking_degraded for r in group),
                        (r.blocking_recon for r in group),
                    ),
                    ringing_retention=spread(_values(group, "ringing_retention")),
                    ringing_pooled=pooled_retention(
                        (r.ringing_clean for r in group),
                        (r.ringing_degraded for r in group),
                        (r.ringing_recon for r in group),
                    ),
                    ringing_gap=spread(r.ringing_degraded - r.ringing_clean for r in group),
                )
            )
    return out


def degraded_source_mismatches(rows: list[PropagationRow]) -> dict[str, tuple[int, float]]:
    """Per source codec: clips whose degraded file differs across VAEs, and the max spread.

    Every VAE should have seen the same degraded file, so
    ``psnr_degraded_clean`` should match to the last digit across VAEs for
    one clip and point. A spread means the file was re-encoded between
    VAE runs and the encoder did not reproduce it bit for bit.
    """
    by_point: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_point[(row.clip_id, row.source_codec, row.source_crf)].append(row.psnr_degraded_clean)
    clips: dict[str, set[str]] = defaultdict(set)
    worst: dict[str, float] = defaultdict(float)
    for (clip, codec, _crf), values in by_point.items():
        width = max(values) - min(values)
        worst[codec] = max(worst[codec], width)
        if width > SAME_FILE_DB:
            clips[codec].add(clip)
    return {codec: (len(clips[codec]), worst[codec]) for codec in _CODEC_ORDER if codec in worst}


def render_summary_csv(summary: list[GroupSummary]) -> str:
    """The summary as CSV, one row per VAE and point, columns in :data:`SUMMARY_COLUMNS`."""
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(SUMMARY_COLUMNS)
    for s in summary:
        writer.writerow(
            [
                s.vae,
                s.source_codec,
                s.source_crf,
                s.clips,
                _num(s.psnr_degraded_clean.median),
                _num(s.psnr_recon_clean.median),
                _num(s.psnr_recon_degraded.median),
                _num(s.clean_recon_psnr),
                _num(s.additive_gap_db),
                _num(s.latent_cosine.median),
                _num(s.latent_relative_l2.median),
                _num(s.latent_relative_l2.q1),
                _num(s.latent_relative_l2.q3),
                _num(s.blocking_retention.median),
                _num(s.blocking_retention.q1),
                _num(s.blocking_retention.q3),
                _num(s.blocking_pooled),
                _num(s.ringing_retention.median),
                _num(s.ringing_retention.q1),
                _num(s.ringing_retention.q3),
                _num(s.ringing_pooled),
                _num(s.ringing_gap.median),
            ]
        )
    return handle.getvalue()


def summary_markdown(summary: list[GroupSummary]) -> str:
    """The table in ``docs/learn/10-artifact-propagation.md``."""
    lines = [
        "| VAE | Source | CRF | PSNR degraded | PSNR recon vs clean | Additive gap | "
        "Latent rel. L2 | Latent cosine | Blocking retention | Ringing retention |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary:
        gap = "" if s.additive_gap_db is None else f"{s.additive_gap_db:+.2f} dB"
        lines.append(
            f"| {s.vae} | {_CODEC_LABEL[s.source_codec]} | {s.source_crf} "
            f"| {s.psnr_degraded_clean.median:.2f} dB | {s.psnr_recon_clean.median:.2f} dB "
            f"| {gap} | {s.latent_relative_l2.median:.3f} | {s.latent_cosine.median:.4f} "
            f"| {_iqr_text(s.blocking_retention)} | {_iqr_text(s.ringing_retention)} |"
        )
    return "\n".join(lines)


def plot_propagation(
    rows: list[PropagationRow],
    summary: list[GroupSummary],
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Three PNGs: pixel PSNR, latent distance, and retention spreads."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    vaes = [v for v in SMOKE_ORDER if any(s.vae == v for s in summary)]
    markers = {"libx264": "o", "libx265": "s", "libsvtav1": "^"}

    def by_vae(
        ax_list: list[Any],
        y: Callable[[GroupSummary], float],
        ylabel: str,
        baseline: bool = False,
    ) -> None:
        for ax, vae in zip(ax_list, vaes, strict=True):
            for codec in _CODEC_ORDER:
                pts = [s for s in summary if s.vae == vae and s.source_codec == codec]
                ax.plot(
                    [s.psnr_degraded_clean.median for s in pts],
                    [y(s) for s in pts],
                    marker=markers[codec],
                    label=_CODEC_LABEL[codec],
                )
            if baseline:
                clean = [s.clean_recon_psnr for s in summary if s.vae == vae and s.clean_recon_psnr]
                if clean:
                    ax.axhline(clean[0], color="grey", linestyle="--", label="clean source")
            ax.set_title(vae)
            ax.set_xlabel("degraded source PSNR-Y (dB)")
            ax.invert_xaxis()
            ax.grid(alpha=0.3)
        ax_list[0].set_ylabel(ylabel)
        ax_list[0].legend(fontsize=8)

    fig, axes = plt.subplots(1, len(vaes), figsize=(4 * len(vaes), 3.6), sharey=True)
    by_vae(list(axes), lambda s: s.psnr_recon_clean.median, "recon vs clean PSNR-Y (dB)", True)
    fig.suptitle("Reconstruction of a compressed source, scored against the clean source (median)")
    fig.tight_layout()
    psnr_path = out_dir / "propagation-psnr.png"
    fig.savefig(psnr_path, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(vaes), figsize=(4 * len(vaes), 3.6), sharey=True)
    by_vae(list(axes), lambda s: s.latent_relative_l2.median, "latent relative L2 to clean")
    fig.suptitle("How far the compressed source moves the latent (median over 34 clips)")
    fig.tight_layout()
    latent_path = out_dir / "propagation-latent.png"
    fig.savefig(latent_path, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    labels = [f"{_CODEC_LABEL[c]}\n{q}" for c, q in study_points()]
    width = 0.8 / len(vaes)
    for ax, column in zip(axes, ("blocking_retention", "ringing_retention"), strict=True):
        for i, vae in enumerate(vaes):
            data = [
                _values(
                    [r for r in rows if (r.vae, r.source_codec, r.source_crf) == (vae, c, q)],
                    column,
                )
                for c, q in study_points()
            ]
            pos = np.arange(len(labels)) + (i - (len(vaes) - 1) / 2) * width
            box = ax.boxplot(
                data,
                positions=pos,
                widths=width * 0.9,
                showfliers=False,
                patch_artist=True,
            )
            color = f"C{i}"
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.plot([], [], color=color, linewidth=6, alpha=0.6, label=vae)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_ylabel(column.replace("_", " ") + "\n(IQR box, whiskers 1.5 IQR)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(ncol=len(vaes), fontsize=8)
    axes[1].set_xticks(np.arange(len(labels)), labels)
    fig.suptitle("Artifact retention per clip: 0 = back to clean, 1 = as damaged as the input")
    fig.tight_layout()
    retention_path = out_dir / "propagation-retention.png"
    fig.savefig(retention_path, dpi=120)
    plt.close(fig)
    return psnr_path, latent_path, retention_path


def _values(rows: list[PropagationRow], column: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = getattr(row, column)
        if value is not None:
            out.append(float(value))
    return out


def _iqr_text(s: Spread) -> str:
    return f"{s.median:.2f} [{s.q1:.2f}, {s.q3:.2f}]"


def _num(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6g}"


__all__ = [
    "SUMMARY_COLUMNS",
    "GroupSummary",
    "Spread",
    "additive_psnr",
    "degraded_source_mismatches",
    "plot_propagation",
    "pooled_retention",
    "render_summary_csv",
    "spread",
    "summarize",
    "summary_markdown",
]
