"""Command-line interface.

``lcb --version`` prints the installed package version. ``lcb data fetch``
downloads the clip manifest and builds the canonical clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from latentcodecbench import __version__
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "data" and args.data_command == "fetch":
        return _fetch(args)
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
