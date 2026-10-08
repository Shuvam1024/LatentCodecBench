"""BD-rate against a hand case and a published PCHIP vector."""

from __future__ import annotations

import pytest

from latentcodecbench.analysis.bdrate import bd_psnr, bd_rate, overlap_fraction

# Four operating points. The test curve is the anchor at half the rate
# and the same distortion, so the log-rate gap is exactly log(1/2).
_RATE = [1000.0, 500.0, 200.0, 80.0]
_PSNR = [42.0, 38.0, 34.0, 30.0]

# Vectors used by the bjontegaard package's own example. method="pchip"
# on that package returns these digits; this implementation is locked to them.
_PUBLISHED_RATE_ANCHOR = [9487.76, 4593.60, 2486.44, 1358.24]
_PUBLISHED_PSNR_ANCHOR = [40.037, 38.615, 36.845, 34.851]
_PUBLISHED_RATE_TEST = [9787.80, 4469.00, 2451.52, 1356.24]
_PUBLISHED_PSNR_TEST = [40.121, 38.651, 36.970, 34.987]


def test_half_the_rate_at_the_same_distortion_is_minus_50_percent() -> None:
    half = [rate / 2 for rate in _RATE]
    # BD-rate is exactly -50%: the distortion ranges coincide and every
    # log-rate is log(1/2) below the anchor. BD-PSNR is not zero. The rate
    # ranges only partly overlap, and on that overlap the test curve sits
    # about 3.18 dB above the anchor. Locked to bjontegaard method="pchip".
    assert bd_rate(_RATE, _PSNR, half, _PSNR) == pytest.approx(-50.0)
    assert bd_psnr(_RATE, _PSNR, half, _PSNR) == pytest.approx(3.1801024775791302)
    assert overlap_fraction(_PSNR, _PSNR) == pytest.approx(1.0)


def test_overlap_fraction_is_intersection_over_union() -> None:
    # Anchor covers 30..42. Test covers 36..48. Intersection is 6, union is 18.
    assert overlap_fraction([42.0, 38.0, 34.0, 30.0], [48.0, 44.0, 40.0, 36.0]) == pytest.approx(
        6.0 / 18.0
    )
    # No shared range.
    assert overlap_fraction([20.0, 18.0, 16.0, 14.0], [40.0, 38.0, 36.0, 34.0]) == pytest.approx(
        0.0
    )


def test_published_pchip_vector() -> None:
    assert bd_rate(
        _PUBLISHED_RATE_ANCHOR,
        _PUBLISHED_PSNR_ANCHOR,
        _PUBLISHED_RATE_TEST,
        _PUBLISHED_PSNR_TEST,
    ) == pytest.approx(-4.417485350589045, rel=1e-9, abs=1e-9)
    assert bd_psnr(
        _PUBLISHED_RATE_ANCHOR,
        _PUBLISHED_PSNR_ANCHOR,
        _PUBLISHED_RATE_TEST,
        _PUBLISHED_PSNR_TEST,
    ) == pytest.approx(0.11969278542479489, rel=1e-9, abs=1e-9)
