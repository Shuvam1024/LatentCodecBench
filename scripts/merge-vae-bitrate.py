#!/usr/bin/env python3
"""Merge an uploaded latent-bitrate CSV into docs/results.

The Colab notebook writes ``/content/results/vae-bitrate.csv``. This script
copies that file into the repo table once it is back on a checkout. A row
already stored under the same clip, VAE, and bit depth is kept. A
conflicting row, or a file hashed from a different config, is refused.

    python scripts/merge-vae-bitrate.py ~/Downloads/vae-bitrate.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latentcodecbench.vaes.bitrate import FULL_SWEEP_ROWS, BitrateError, merge_bitrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge a VAE bitrate CSV into docs/results.")
    parser.add_argument("csv", type=Path, help="CSV downloaded from Colab.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("docs/results/vae-bitrate.csv"),
        help="Repo table to update.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae-bitrate.yaml"),
        help="Config whose SHA256 the CSV must match.",
    )
    args = parser.parse_args(argv)
    try:
        rows = merge_bitrate(args.csv, args.dest, config_path=args.config)
    except (BitrateError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} rows to {args.dest}")
    if len(rows) != FULL_SWEEP_ROWS:
        print(
            f"a full sweep is {FULL_SWEEP_ROWS} rows "
            f"(34 clips x 4 VAEs x 6 bit depths); this file has {len(rows)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
