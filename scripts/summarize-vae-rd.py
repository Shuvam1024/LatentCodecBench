#!/usr/bin/env python3
"""Print the VAE BD-rate table, and optionally write the CSV and figures.

    python scripts/summarize-vae-rd.py
    python scripts/summarize-vae-rd.py --write

The markdown is the table in ``docs/learn/09-vae-rd-curves.md``. A test
checks that the note and ``docs/results/vae-rd-bdrate.csv`` still match
this output. ADR-0011.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latentcodecbench.analysis.table import read_results
from latentcodecbench.analysis.vae_rd import (
    plot_vae_rd,
    render_bdrate_csv,
    vae_bd_rate_markdown,
    vae_clip_bd_rates,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize VAE BD-rate against the classics.")
    parser.add_argument(
        "--classic",
        type=Path,
        default=Path("docs/results/classic-baselines.csv"),
    )
    parser.add_argument(
        "--bitrate",
        type=Path,
        default=Path("docs/results/vae-bitrate.csv"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite docs/results/vae-rd-bdrate.csv and the three figures.",
    )
    args = parser.parse_args(argv)
    try:
        rows = read_results(args.classic) + read_results(args.bitrate)
        print(vae_bd_rate_markdown(rows))
        if args.write:
            clips = vae_clip_bd_rates(rows)
            dest = Path("docs/results/vae-rd-bdrate.csv")
            dest.write_text(render_bdrate_csv(clips), encoding="utf-8")
            psnr, vmaf, per_clip = plot_vae_rd(rows, Path("docs/results"))
            print(dest)
            print(psnr)
            print(vmaf)
            print(per_clip)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
