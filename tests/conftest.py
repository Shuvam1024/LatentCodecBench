"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from latentcodecbench.testing import make_synthetic_clip


@pytest.fixture
def synthetic_clip() -> npt.NDArray[np.uint8]:
    """A small deterministic clip for tests that need video-shaped input."""
    return make_synthetic_clip()
