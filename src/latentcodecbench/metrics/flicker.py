"""Temporal flicker: how frame-to-frame change disagrees.

For a reference clip ``R`` and a distorted clip ``D``,

    mean | (R[t] - R[t - 1]) - (D[t] - D[t - 1]) |

over every sample and every pair of neighboring frames. Identical clips
score 0. A constant offset added to every frame also scores 0, because
the deltas did not change. A frame that jumps when the reference does
not scores high.

This is a stand-in for optical-flow warping error. Warping would ask
whether the distorted motion matches the reference motion, which needs
a flow field per pair of frames. That is left as future work: it is a
heavy dependency for a CPU baseline, and this delta already catches
flicker, frozen frames, and temporal pumping.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def temporal_flicker(
    reference: npt.NDArray[np.generic],
    distorted: npt.NDArray[np.generic],
) -> float:
    """Mean absolute disagreement of frame-to-frame deltas."""
    if reference.shape != distorted.shape:
        raise ValueError(f"shape {reference.shape} != {distorted.shape}")
    if reference.shape[0] < 2:
        raise ValueError("flicker needs at least two frames")
    ref = reference.astype(np.float64)
    dist = distorted.astype(np.float64)
    disagreement = (ref[1:] - ref[:-1]) - (dist[1:] - dist[:-1])
    return float(np.mean(np.abs(disagreement)))
