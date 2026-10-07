"""Command-line interface.

``lcb --version`` prints the installed package version. ``lcb data fetch``
downloads the clip manifest and builds the canonical clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from latentcodecbench import __version__
from latentcodecbench.analysis.plot import plot_rd
from latentcodecbench.analysis.run import RunError, run_baselines
from latentcodecbench.analysis.table import read_results
from latentcodecbench.data.fetch import HashMismatchError, fetch_clip
from latentcodecbench.data.manifest import load_manifest
from latentcodecbench.data.preprocess import PreprocessError, preprocess_clip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcb",
        description="Benchmark open video VAEs against classic video codecs.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"latentcodecbench {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=False)
    data = sub.add_parser("data", help="Clip manifest, download, and preprocessing.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="Download sources and build canonical clips.")
    fetch.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/clips.yaml"),
        help="Path to the clip manifest.",
    )
    fetch.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    fetch.add_argument("--clip-dir", type=Path, default=Path("data/clips"))
    fetch.add_argument(
        "--only",
        action="append",
        default=None,
        help="Fetch a single clip id. Repeat to fetch several.",
    )
    baselines = sub.add_parser("baselines", help="Classic codec sweeps.")
    baselines_sub = baselines.add_subparsers(dest="baselines_command", required=True)
    run = baselines_sub.add_parser("run", help="Encode the clip set and write a results table.")
    run.add_argument("--config", type=Path, default=Path("configs/classic.yaml"))
    run.add_argument("--manifest", type=Path, default=Path("manifests/clips.yaml"))
    run.add_argument("--clip-dir", type=Path, default=Path("data/clips"))
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, default=Path("results/work"))
    run.add_argument("--only", action="append", default=None)
    plot = sub.add_parser("plot", help="Figures from a results table.")
    plot_sub = plot.add_subparsers(dest="plot_command", required=True)
    rd = plot_sub.add_parser("rd", help="Rate-distortion curves and mean BD-rate.")
    rd.add_argument("--csv", type=Path, required=True)
    rd.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    rd.add_argument("--anchor", default="libx264")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "data" and args.data_command == "fetch":
        return _fetch(args)
    if args.command == "baselines" and args.baselines_command == "run":
        return _baselines_run(args)
    if args.command == "plot" and args.plot_command == "rd":
        return _plot_rd(args)
    parser.print_help()
    return 2


def _fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    wanted = set(args.only) if args.only else None
    if wanted is not None:
        known = {clip.id for clip in manifest.clips}
        missing = wanted - known
        if missing:
            print(f"unknown clip id: {', '.join(sorted(missing))}")
            return 2
    failures = 0
    for clip in manifest.clips:
        if wanted is not None and clip.id not in wanted:
            continue
        try:
            source = fetch_clip(clip, args.raw_dir)
            destination = preprocess_clip(clip, source, args.clip_dir)
        except (HashMismatchError, PreprocessError, OSError, RuntimeError) as exc:
            print(exc)
            failures += 1
            continue
        print(f"{clip.id}: {destination}")
    return 1 if failures else 0


def _baselines_run(args: argparse.Namespace) -> int:
    try:
        rows = run_baselines(
            manifest_path=args.manifest,
            config_path=args.config,
            clip_dir=args.clip_dir,
            out=args.out,
            work_dir=args.work_dir,
            only=set(args.only) if args.only else None,
        )
    except (RunError, OSError, RuntimeError) as exc:
        print(exc)
        return 1
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


def _plot_rd(args: argparse.Namespace) -> int:
    rows = read_results(args.csv)
    if not rows:
        print(f"{args.csv} has no rows")
        return 1
    psnr_path, vmaf_path = plot_rd(rows, args.out_dir, anchor=args.anchor)
    print(psnr_path)
    print(vmaf_path)
    return 0
