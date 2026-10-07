"""Command-line interface.

``lcb --version`` prints the installed package version. Subcommands for
running benchmarks arrive with the phases that implement them.
"""

from __future__ import annotations

import argparse

from latentcodecbench import __version__


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
