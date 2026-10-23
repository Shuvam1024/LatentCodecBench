#!/usr/bin/env python3
"""Print the artifact-propagation table, and optionally write the CSV and figures.

    python scripts/summarize-artifact-propagation.py
    python scripts/summarize-artifact-propagation.py --write

The markdown is the table in ``docs/learn/10-artifact-propagation.md``.
``--write`` rewrites ``docs/results/artifact-propagation-summary.csv`` and
the three ``docs/results/propagation-*.png`` figures. ADR-0012.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latentcodecbench.analysis.propagation import read_propagation
from latentcodecbench.analysis.propagation_summary import (
    degraded_source_mismatches,
    plot_propagation,
    render_summary_csv,
    summarize,
    summary_markdown,
)
from latentcodecbench.analysis.table import read_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the artifact-propagation sweep.")
    parser.add_argument(
        "--results", type=Path, default=Path("docs/results/artifact-propagation.csv")
    )
    parser.add_argument("--recon", type=Path, default=Path("docs/results/vae-recon.csv"))
    parser.add_argument("--write", action="store_true", help="Rewrite the summary CSV and figures.")
    args = parser.parse_args(argv)
    try:
        rows = read_propagation(args.results)
        summary = summarize(rows, read_results(args.recon))
        print(summary_markdown(summary))
        print()
        for codec, (clips, worst) in degraded_source_mismatches(rows).items():
            print(
                f"{codec}: {clips} clips differ across VAEs, max {worst:.4f} dB"
            )
        if args.write:
            dest = Path("docs/results/artifact-propagation-summary.csv")
            dest.write_text(render_summary_csv(summary), encoding="utf-8")
            print(dest)
            for path in plot_propagation(rows, summary, Path("docs/results")):
                print(path)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
