#!/usr/bin/env python3
"""Re-encode canonical clips at the Phase 3 study ladder.

    python scripts/degrade-sources.py
    python scripts/degrade-sources.py --clip-dir data/clips --out-dir data/degraded
    python scripts/degrade-sources.py --only aspen --only crowd_run

Nine files per clip: x264 and x265 at CRF 18, 30, 38, and SVT-AV1 at CRF
28, 49, 63. Presets are the Phase 1 benchmark presets. A y4m that already
matches the source geometry is skipped. Output stays under ``data/``,
which is gitignored. ADR-0012.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latentcodecbench.codecs.baseline import EncodeError
from latentcodecbench.codecs.degrade import degrade_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write degraded y4m sources for Phase 3.")
    parser.add_argument("--clip-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/degraded"))
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Clip id to include. Repeat for more than one. Default: every y4m.",
    )
    args = parser.parse_args(argv)
    only = set(args.only) if args.only else None
    try:
        written = degrade_sources(args.clip_dir, args.out_dir, only=only)
    except EncodeError as exc:
        print(exc, file=sys.stderr)
        return 1
    fresh = 0
    for row in written:
        if row.skipped:
            print(f"skip {row.y4m}")
            continue
        fresh += 1
        print(
            f"wrote {row.y4m} encode={row.encode_seconds:.2f}s "
            f"decode={row.pixel_decode_seconds:.2f}s"
        )
    print(f"{fresh} new, {len(written) - fresh} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
