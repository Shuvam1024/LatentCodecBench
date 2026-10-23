"""Check the published pages against the committed CSVs.

``lcb reproduce`` rebuilds the headline, the BD-rate table, the
propagation table, and the training-filter section from those CSVs. It
fails when a page no longer contains that text, when a derived CSV no
longer matches its renderer, or when a figure is missing. It redraws the
figures in a temporary directory. It does not replace the committed PNGs,
and it does not re-run a codec or a VAE.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from latentcodecbench.analysis.bdrate import BjontegaardError
from latentcodecbench.analysis.filtering import filtering_report
from latentcodecbench.analysis.plot import mean_bd_rate, plot_rd
from latentcodecbench.analysis.propagation import PropagationRow, read_propagation
from latentcodecbench.analysis.propagation_summary import (
    plot_propagation,
    render_summary_csv,
    summarize,
    summary_markdown,
)
from latentcodecbench.analysis.report_tables import recon_interval_markdown, report_numbers
from latentcodecbench.analysis.table import ResultRow, read_results
from latentcodecbench.analysis.vae_rd import (
    DISPLAY,
    METRIC_LABEL,
    plot_vae_rd,
    render_bdrate_csv,
    vae_bd_rate_markdown,
    vae_bd_rate_means,
    vae_clip_bd_rates,
)
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

_FIGURES: tuple[str, ...] = (
    "docs/figures/rd-psnr.png",
    "docs/figures/rd-vmaf.png",
    "docs/results/rd-psnr.png",
    "docs/results/rd-vmaf.png",
    "docs/results/rd-bdrate-clips.png",
    "docs/results/propagation-psnr.png",
    "docs/results/propagation-latent.png",
    "docs/results/propagation-retention.png",
)
_MIN_FIGURE_BYTES = 1000


@dataclass(frozen=True)
class PublishedTables:
    """The committed result tables, loaded once for the check and the headline."""

    classic: list[ResultRow]
    recon: list[ResultRow]
    bitrate: list[ResultRow]
    propagation: list[PropagationRow]


def load_published(root: Path) -> PublishedTables:
    """Read the four CSVs the published pages are built from."""
    results = root / "docs" / "results"
    return PublishedTables(
        classic=read_results(results / "classic-baselines.csv"),
        recon=read_results(results / "vae-recon.csv"),
        bitrate=read_results(results / "vae-bitrate.csv"),
        propagation=read_propagation(results / "artifact-propagation.csv"),
    )


def headline_markdown(tables: PublishedTables) -> str:
    """The short results table in the README and in note 12."""
    lines = [
        "| Codec | Against | Metric | BD-rate | Clips in the mean | Mean overlap | Under 75% |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in ("psnr_y", "vmaf"):
        for summary in mean_bd_rate(tables.classic, metric=metric):
            lines.append(
                f"| {DISPLAY.get(summary.codec, summary.codec)} | x264 "
                f"| {METRIC_LABEL[metric]} | {summary.mean_bd_rate:+.2f}% "
                f"| {summary.clips} | {summary.mean_overlap * 100:.2f}% "
                f"| {summary.below_overlap} |"
            )
    combined = tables.classic + tables.bitrate
    for point in vae_bd_rate_means(vae_clip_bd_rates(combined)):
        if point.anchor != "libx264" or point.mean_bd_rate is None or point.mean_overlap is None:
            continue
        lines.append(
            f"| {point.codec} | x264 | {METRIC_LABEL[point.metric]} "
            f"| {point.mean_bd_rate:+.2f}% | {point.defined} "
            f"| {point.mean_overlap * 100:.2f}% | {point.below_overlap} |"
        )
    lines.extend(["", _recon_table(tables.recon)])
    return "\n".join(lines)


def check_published(root: Path) -> list[str]:
    """Problems with the published pages. An empty list means they match."""
    problems: list[str] = []
    try:
        tables = load_published(root)
    except (OSError, ValueError, BjontegaardError) as exc:
        return [str(exc)]
    if not (tables.classic and tables.recon and tables.bitrate and tables.propagation):
        return ["one or more committed result CSVs are missing or empty"]
    headline = headline_markdown(tables)
    combined = tables.classic + tables.bitrate
    expected = {
        root / "README.md": headline,
        root / "docs" / "learn" / "12-reproducing-the-tables.md": headline,
        root / "docs" / "learn" / "09-vae-rd-curves.md": vae_bd_rate_markdown(combined),
        root / "docs" / "learn" / "10-artifact-propagation.md": summary_markdown(
            summarize(tables.propagation, tables.recon)
        ),
        root / "docs" / "learn" / "11-training-data-filter.md": filtering_report(
            tables.propagation, tables.recon
        ),
        root / "docs" / "REPORT.md": report_numbers(
            tables.classic, tables.recon, tables.bitrate, tables.propagation
        ),
        root / "docs" / "learn" / "13-confidence-intervals.md": recon_interval_markdown(
            tables.recon
        ),
    }
    for path, text in expected.items():
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        if text not in path.read_text(encoding="utf-8"):
            problems.append(f"{path} does not contain the table rebuilt from the CSVs")
    problems.extend(_check_derived_csv(root, tables))
    problems.extend(_check_figures(root))
    problems.extend(_redraw(tables))
    return problems


def _recon_table(rows: list[ResultRow]) -> str:
    lines = [
        "| VAE | Clips | Mean PSNR-Y | Mean VMAF |",
        "| --- | ---: | ---: | ---: |",
    ]
    grouped: dict[str, list[ResultRow]] = {name: [] for name in SMOKE_ORDER}
    for row in rows:
        grouped.setdefault(row.codec, []).append(row)
    for name in SMOKE_ORDER:
        group = grouped.get(name, [])
        if not group:
            continue
        lines.append(
            f"| {name} | {len(group)} | {fmean(row.psnr_y for row in group):.2f} dB "
            f"| {fmean(row.vmaf for row in group):.2f} |"
        )
    return "\n".join(lines)


def _check_derived_csv(root: Path, tables: PublishedTables) -> list[str]:
    problems: list[str] = []
    results = root / "docs" / "results"
    bdrate = render_bdrate_csv(vae_clip_bd_rates(tables.classic + tables.bitrate))
    summary = render_summary_csv(summarize(tables.propagation, tables.recon))
    checks = {
        results / "vae-rd-bdrate.csv": bdrate,
        results / "artifact-propagation-summary.csv": summary,
    }
    for path, text in checks.items():
        if not path.is_file():
            problems.append(f"missing {path}")
        elif path.read_text(encoding="utf-8") != text:
            problems.append(f"{path.name} does not match the renderer")
    return problems


def _check_figures(root: Path) -> list[str]:
    problems: list[str] = []
    for relative in _FIGURES:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= _MIN_FIGURE_BYTES:
            problems.append(f"missing or empty figure {relative}")
    return problems


def _redraw(tables: PublishedTables) -> list[str]:
    """Draw the figures from the CSVs into a temporary directory."""
    try:
        with TemporaryDirectory() as directory:
            out = Path(directory)
            with redirect_stdout(io.StringIO()):
                plot_rd(tables.classic, out / "classic")
                plot_vae_rd(tables.classic + tables.bitrate, out / "vae")
                plot_propagation(
                    tables.propagation,
                    summarize(tables.propagation, tables.recon),
                    out / "propagation",
                )
            drawn = list(out.rglob("*.png"))
            if len(drawn) < 8 or any(path.stat().st_size <= _MIN_FIGURE_BYTES for path in drawn):
                return ["redrawing the figures did not write eight non-empty PNGs"]
    except (OSError, ValueError, BjontegaardError) as exc:
        return [f"redraw failed: {exc}"]
    return []
