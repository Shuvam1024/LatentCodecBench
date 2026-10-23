"""Student-t intervals for a mean across clips.

The clips are a designed set, not a draw from all video. The interval is
the clip-to-clip spread of the mean: where that mean would fall if these
clips were a sample from a larger population of the same kind. It is not
the uncertainty of one forward. One row is deterministic given the
weights, the dtype, and the file.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, stdev

from scipy.stats import t as student_t

from latentcodecbench.analysis.bdrate import BjontegaardError, bd_rate
from latentcodecbench.analysis.table import ResultRow
from latentcodecbench.analysis.vae_rd import vae_clip_bd_rates

_CLASSIC_ORDER = ("libx265", "libsvtav1")
_ANCHOR = "libx264"


@dataclass(frozen=True)
class MeanInterval:
    """Mean and a two-sided Student-t interval. Bounds are NaN when n < 2."""

    n: int
    mean: float
    low: float
    high: float


def student_t_interval(values: list[float], *, alpha: float = 0.05) -> MeanInterval:
    """Two-sided Student-t interval for the mean. ``alpha`` is the tail mass.

    Non-finite values are dropped. An empty list raises ``ValueError``.
    """
    data = [value for value in values if math.isfinite(value)]
    if not data:
        raise ValueError("no values")
    n = len(data)
    mean = fmean(data)
    if n < 2:
        nan = float("nan")
        return MeanInterval(n, mean, nan, nan)
    scale = stdev(data) / math.sqrt(n)
    critical = float(student_t.ppf(1 - alpha / 2, n - 1))
    half = critical * scale
    return MeanInterval(n, mean, mean - half, mean + half)


def classic_bd_rate_samples(
    rows: list[ResultRow],
    *,
    metric: str,
    anchor: str = _ANCHOR,
) -> dict[str, list[float]]:
    """Per-clip BD-rates that enter :func:`mean_bd_rate`, in codec order.

    A clip with fewer than four points, or a non-monotonic curve, is
    absent. The lists are the samples a confidence interval may use.
    """
    by_clip: dict[str, dict[str, list[ResultRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_clip[row.clip_id][row.codec].append(row)
    found: dict[str, list[float]] = {codec: [] for codec in _CLASSIC_ORDER}
    for codec in _CLASSIC_ORDER:
        for clip_id in sorted(by_clip):
            anchor_rows = by_clip[clip_id].get(anchor, [])
            test_rows = by_clip[clip_id].get(codec, [])
            if len(anchor_rows) < 4 or len(test_rows) < 4:
                continue
            anchor_dist = [float(getattr(row, metric)) for row in anchor_rows]
            test_dist = [float(getattr(row, metric)) for row in test_rows]
            try:
                found[codec].append(
                    bd_rate(
                        [_bpp(row) for row in anchor_rows],
                        anchor_dist,
                        [_bpp(row) for row in test_rows],
                        test_dist,
                    )
                )
            except BjontegaardError:
                continue
    return found


def vae_bd_rate_samples(
    rows: list[ResultRow],
    *,
    anchor: str = _ANCHOR,
) -> dict[tuple[str, str], list[float]]:
    """Defined per-clip VAE BD-rates against ``anchor``, keyed by codec and metric."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in vae_clip_bd_rates(rows):
        if row.anchor != anchor or row.status != "defined" or row.bd_rate is None:
            continue
        grouped[(row.codec, row.metric)].append(row.bd_rate)
    return grouped


def _bpp(row: ResultRow) -> float:
    if row.bpp is None:
        raise BjontegaardError(f"{row.clip_id} {row.codec} has no bits per pixel")
    return row.bpp
