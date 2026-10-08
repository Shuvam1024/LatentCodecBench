"""Bjontegaard deltas with a piecewise-cubic interpolant.

BD-rate is the average bitrate change, in percent, over the overlap of
the two distortion ranges. The interpolant is a PCHIP of ``log(rate)``
against distortion, which is the curve the JCT-VC common test conditions
use and the one ``bjontegaard`` (method ``pchip``) computes. A polynomial
fit through four points can swing outside the samples; PCHIP does not.

BD-PSNR is the same idea with the axes swapped: average distortion
change, in the distortion unit, over the overlap of the log-rate ranges.

A negative BD-rate means the test codec spends fewer bits than the anchor
at the same distortion.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


class BjontegaardError(ValueError):
    """The two curves do not define a BD delta."""


# ``bjontegaard`` warns when this fraction is smaller. The fraction is the
# intersection of the two ranges divided by their union.
OVERLAP_FLAG = 0.75


def overlap_fraction(
    distortion_anchor: list[float] | tuple[float, ...],
    distortion_test: list[float] | tuple[float, ...],
) -> float:
    """Shared fraction of two distortion ranges, in ``[0, 1]``.

    BD-rate integrates over the intersection. This number is how much of
    the combined range that intersection is. A value under
    ``OVERLAP_FLAG`` is the case ``bjontegaard`` warns about.
    """
    anchor = np.asarray(distortion_anchor, dtype=np.float64)
    test = np.asarray(distortion_test, dtype=np.float64)
    if anchor.size == 0 or test.size == 0:
        return 0.0
    total = float(max(np.max(anchor), np.max(test)) - min(np.min(anchor), np.min(test)))
    if total <= 0.0:
        return 0.0
    low = float(max(np.min(anchor), np.min(test)))
    high = float(min(np.max(anchor), np.max(test)))
    return max(high - low, 0.0) / total


def bd_rate(
    rate_anchor: list[float] | tuple[float, ...],
    distortion_anchor: list[float] | tuple[float, ...],
    rate_test: list[float] | tuple[float, ...],
    distortion_test: list[float] | tuple[float, ...],
) -> float:
    """BD-rate of the test curve against the anchor, in percent."""
    anchor_rate, anchor_dist = _as_curve(rate_anchor, distortion_anchor, "anchor")
    test_rate, test_dist = _as_curve(rate_test, distortion_test, "test")
    low = float(max(np.min(anchor_dist), np.min(test_dist)))
    high = float(min(np.max(anchor_dist), np.max(test_dist)))
    if high - low <= 1e-9:
        raise BjontegaardError("distortion ranges do not overlap")
    anchor_log = _interpolant(anchor_dist, np.log(anchor_rate))
    test_log = _interpolant(test_dist, np.log(test_rate))
    average = (test_log.integrate(low, high) - anchor_log.integrate(low, high)) / (high - low)
    return float(np.exp(average) - 1.0) * 100.0


def bd_psnr(
    rate_anchor: list[float] | tuple[float, ...],
    distortion_anchor: list[float] | tuple[float, ...],
    rate_test: list[float] | tuple[float, ...],
    distortion_test: list[float] | tuple[float, ...],
) -> float:
    """BD-PSNR of the test curve against the anchor, in distortion units."""
    anchor_rate, anchor_dist = _as_curve(rate_anchor, distortion_anchor, "anchor")
    test_rate, test_dist = _as_curve(rate_test, distortion_test, "test")
    anchor_log = np.log(anchor_rate)
    test_log = np.log(test_rate)
    low = float(max(np.min(anchor_log), np.min(test_log)))
    high = float(min(np.max(anchor_log), np.max(test_log)))
    if high - low <= 1e-12:
        raise BjontegaardError("rate ranges do not overlap")
    anchor_dist_of_rate = _interpolant(anchor_log, anchor_dist)
    test_dist_of_rate = _interpolant(test_log, test_dist)
    average = (
        test_dist_of_rate.integrate(low, high) - anchor_dist_of_rate.integrate(low, high)
    ) / (high - low)
    return float(average)


def _as_curve(
    rate: list[float] | tuple[float, ...],
    distortion: list[float] | tuple[float, ...],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rate) != len(distortion) or len(rate) < 4:
        raise BjontegaardError(f"{name} needs at least 4 paired rate/distortion samples")
    rate_array = np.asarray(rate, dtype=np.float64)
    distortion_array = np.asarray(distortion, dtype=np.float64)
    if np.any(rate_array <= 0):
        raise BjontegaardError(f"{name} rates must be positive")
    return rate_array, distortion_array


def _interpolant(x_values: np.ndarray, y_values: np.ndarray) -> PchipInterpolator:
    order = np.argsort(x_values)
    ordered_x = x_values[order]
    if np.any(np.diff(ordered_x) <= 0):
        raise BjontegaardError("distortion or log-rate samples are not strictly monotonic")
    return PchipInterpolator(ordered_x, y_values[order])
