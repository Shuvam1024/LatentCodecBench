"""Fixed-length features for the source-quality probe.

A posterior mean is a different shape for each VAE, and the raw tensor
is about a megabyte. The probe stores channel statistics instead: mean,
standard deviation, and mean absolute value, padded out to 128 channels,
plus one high-frequency energy. The same block is stored for the
difference from the clean posterior mean. A clean row stores that
difference as zeros.

Pixel features are a short luma summary of the y4m that was encoded.
They are the baseline that does not look at the latent. ADR-0013.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from latentcodecbench.metrics.artifacts import blocking_score, latent_distance, ringing_score

# LTX-Video is 128 channels. Wan2.1 and Hunyuan are 16, Wan2.2 is 48.
# Padding keeps one CSV header for every VAE. A column that is zero for
# a whole VAE has no variance and the fit drops it.
MAX_LATENT_CHANNELS = 128

PIXEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "pixel_luma_mean",
    "pixel_luma_std",
    "pixel_blocking",
    "pixel_ringing",
    "pixel_hf_energy",
    "pixel_temporal_energy",
)

_DISTANCE_COLUMNS: tuple[str, ...] = (
    "delta_mse",
    "delta_relative_l2",
    "delta_cosine",
)


def _stat_names(prefix: str) -> tuple[str, ...]:
    names = [
        f"{prefix}_{stat}_c{index:03d}"
        for stat in ("mean", "std", "abs_mean")
        for index in range(MAX_LATENT_CHANNELS)
    ]
    names.append(f"{prefix}_hf_energy")
    return tuple(names)


LATENT_FEATURE_COLUMNS: tuple[str, ...] = (
    _stat_names("latent") + _stat_names("delta") + _DISTANCE_COLUMNS
)
FEATURE_COLUMNS: tuple[str, ...] = LATENT_FEATURE_COLUMNS + PIXEL_FEATURE_COLUMNS


def as_volume(latent: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    """``(channels, time, height, width)`` from a posterior mean.

    A leading batch of 1 is removed. Any other rank is an error.
    """
    array = np.asarray(latent, dtype=np.float64)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError(f"latent batch {array.shape[0]} is not 1")
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"latent must be (channels, time, height, width), got {array.shape}")
    if array.size == 0:
        raise ValueError("latent is empty")
    return array


def probe_feature_vector(
    latent: npt.NDArray[np.generic],
    clean: npt.NDArray[np.generic],
    luma: npt.NDArray[np.generic],
) -> npt.NDArray[np.float64]:
    """One row of :data:`FEATURE_COLUMNS` for ``latent`` against ``clean``."""
    current = as_volume(latent)
    reference = as_volume(clean)
    if current.shape != reference.shape:
        raise ValueError(f"latent shape {current.shape} != clean {reference.shape}")
    distance = latent_distance(reference, current)
    tail = np.array(
        [distance.mse, distance.relative_l2, distance.cosine],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            _channel_block(current),
            _channel_block(current - reference),
            tail,
            pixel_feature_vector(luma),
        ]
    )


def pixel_feature_vector(luma: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    """Luma mean, spread, blocking, ringing, spatial highpass, and frame step.

    ``luma`` is ``(frames, height, width)`` in 0-255. Blocking needs both
    sides divisible by 8 and at least 16. High-frequency energy is the
    mean square of the horizontal neighbor difference. Temporal energy is
    the mean absolute frame-to-frame difference, or 0 for one frame.
    """
    if luma.ndim != 3:
        raise ValueError(f"luma must be (frames, height, width), got {luma.shape}")
    video = luma.astype(np.float64, copy=False)
    if video.shape[-1] >= 2:
        spatial = np.diff(video, axis=-1)
        high = float(np.mean(spatial * spatial))
    else:
        high = 0.0
    temporal = float(np.mean(np.abs(video[1:] - video[:-1]))) if video.shape[0] >= 2 else 0.0
    flat = video.reshape(-1)
    return np.array(
        [
            float(flat.mean()),
            float(flat.std()),
            blocking_score(luma),
            ringing_score(luma),
            high,
            temporal,
        ],
        dtype=np.float64,
    )


def _channel_block(volume: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    channels = int(volume.shape[0])
    if channels > MAX_LATENT_CHANNELS:
        raise ValueError(f"{channels} latent channels exceeds {MAX_LATENT_CHANNELS}")
    flat = volume.reshape(channels, -1)
    block = np.zeros(MAX_LATENT_CHANNELS * 3 + 1, dtype=np.float64)
    block[0:channels] = flat.mean(axis=1)
    start = MAX_LATENT_CHANNELS
    block[start : start + channels] = flat.std(axis=1)
    start = 2 * MAX_LATENT_CHANNELS
    block[start : start + channels] = np.abs(flat).mean(axis=1)
    block[-1] = _hf_energy(volume)
    return block


def _hf_energy(volume: npt.NDArray[np.float64]) -> float:
    """Mean square of the first difference along width, else along height."""
    if volume.shape[-1] >= 2:
        diff = np.diff(volume, axis=-1)
    elif volume.shape[-2] >= 2:
        diff = np.diff(volume, axis=-2)
    else:
        return 0.0
    return float(np.mean(diff * diff))
