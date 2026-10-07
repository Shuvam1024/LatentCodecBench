"""Tests for the synthetic clip helper."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from latentcodecbench.testing import make_synthetic_clip


def test_default_shape_and_dtype(synthetic_clip: npt.NDArray[np.uint8]) -> None:
    assert synthetic_clip.shape == (8, 32, 32, 3)
    assert synthetic_clip.dtype == np.uint8


def test_seed_is_deterministic() -> None:
    first = make_synthetic_clip(seed=7)
    second = make_synthetic_clip(seed=7)
    other = make_synthetic_clip(seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_motion_changes_a_later_frame() -> None:
    clip = make_synthetic_clip(num_frames=4, height=16, width=16, seed=1)
    assert not np.array_equal(clip[0], clip[3])


def test_custom_shape() -> None:
    clip = make_synthetic_clip(num_frames=2, height=4, width=6, num_channels=1, seed=0)
    assert clip.shape == (2, 4, 6, 1)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"num_frames": 0}, "num_frames"),
        ({"height": -1}, "height"),
        ({"width": 0}, "width"),
        ({"num_channels": 0}, "num_channels"),
    ],
)
def test_non_positive_dimensions_raise(kwargs: dict[str, int], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_synthetic_clip(**kwargs)
