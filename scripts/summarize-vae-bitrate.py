#!/usr/bin/env python3
"""Print mean PSNR-Y, mean VMAF, and median bpp for the bitrate CSV.

    python scripts/summarize-vae-bitrate.py
    python scripts/summarize-vae-bitrate.py docs/results/vae-bitrate.csv

The table is the one in ``docs/learn/05-latent-bitrate.md``. A test checks
that the note still contains this output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latentcodecbench.analysis.bitrate_table import bitrate_summary_markdown
from latentcodecbench.analysis.table import read_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a VAE bitrate CSV.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("docs/results/vae-bitrate.csv"),
        help="Bitrate results table.",
    )
    args = parser.parse_args(argv)
    try:
        print(bitrate_summary_markdown(read_results(args.csv)))
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
