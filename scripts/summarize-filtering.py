#!/usr/bin/env python3
"""Print the training-filter table from the committed propagation rows.

    python scripts/summarize-filtering.py

The markdown is the measured section of ``docs/learn/11-training-data-filter.md``.
A test checks that the note still contains this printout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latentcodecbench.analysis.filtering import filtering_report
from latentcodecbench.analysis.propagation import read_propagation
from latentcodecbench.analysis.table import read_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the training-data filter.")
    parser.add_argument(
        "--propagation",
        type=Path,
        default=Path("docs/results/artifact-propagation.csv"),
    )
    parser.add_argument("--recon", type=Path, default=Path("docs/results/vae-recon.csv"))
    args = parser.parse_args(argv)
    try:
        print(filtering_report(read_propagation(args.propagation), read_results(args.recon)))
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
