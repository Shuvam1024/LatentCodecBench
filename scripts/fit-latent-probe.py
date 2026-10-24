#!/usr/bin/env python3
"""Fit the source-CRF probe from a feature CSV.

    python scripts/fit-latent-probe.py --features features.csv --out out.csv

``--note`` replaces the marked section of a learn note. The same command
is ``lcb probe fit``.
"""

from __future__ import annotations

import sys

from latentcodecbench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["probe", "fit", *sys.argv[1:]]))
