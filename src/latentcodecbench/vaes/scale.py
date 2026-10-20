"""Published latent normalization for the four pinned VAEs.

The numbers are the ones the Diffusers 0.41.0 pipelines apply before a
diffusion model sees the posterior mean, and the inverse of that step
before ``decode``. Wan stores a per-channel mean and standard deviation
on ``vae/config.json``. HunyuanVideo stores one ``scaling_factor``.
LTX-Video stores ``scaling_factor`` in the config and the per-channel
mean and standard deviation in the VAE weights. See ADR-0010.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np
import numpy.typing as npt

from latentcodecbench.vaes.spec import GEOMETRIES


class ScaleError(ValueError):
    """The published scale does not match the latent."""


@dataclass(frozen=True)
class LatentScale:
    """Affine map ``(latent - shift) * scale`` into normalized space.

    ``shift`` and ``scale`` each have one entry per latent channel.
    """

    name: str
    shift: tuple[float, ...]
    scale: tuple[float, ...]

    def shift_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.shift, dtype=np.float64).reshape(-1, 1, 1, 1)

    def scale_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.scale, dtype=np.float64).reshape(-1, 1, 1, 1)


@lru_cache(maxsize=1)
def load_scales() -> dict[str, LatentScale]:
    """Read the packaged scale table once."""
    raw = files("latentcodecbench.vaes").joinpath("latent_scale.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    if payload.get("version") != 1:
        raise ScaleError("latent_scale.json version is not 1")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ScaleError("latent_scale.json has no models")
    scales: dict[str, LatentScale] = {}
    for name, spec in models.items():
        if name not in GEOMETRIES:
            raise ScaleError(f"unknown VAE {name}")
        shift = tuple(float(value) for value in spec["shift"])
        scale = tuple(float(value) for value in spec["scale"])
        expected = GEOMETRIES[name].latent_channels
        if len(shift) != expected or len(scale) != expected:
            raise ScaleError(f"{name} scale length != {expected} channels")
        if any(value == 0.0 for value in scale):
            raise ScaleError(f"{name} has a zero scale")
        scales[name] = LatentScale(name=name, shift=shift, scale=scale)
    missing = [name for name in GEOMETRIES if name not in scales]
    if missing:
        raise ScaleError(f"latent scale missing {', '.join(missing)}")
    return scales


def published_scale(name: str) -> LatentScale:
    """The pinned normalization for one VAE name."""
    try:
        return load_scales()[name]
    except KeyError as exc:
        raise ScaleError(f"no published scale for {name}") from exc


def normalize(latent: npt.NDArray[np.float64], scale: LatentScale) -> npt.NDArray[np.float64]:
    """Map a raw posterior mean into the space the quantizer sees."""
    _check_channels(latent, scale)
    return (latent - scale.shift_array()) * scale.scale_array()


def denormalize(latent: npt.NDArray[np.float64], scale: LatentScale) -> npt.NDArray[np.float64]:
    """Invert :func:`normalize`."""
    _check_channels(latent, scale)
    return latent / scale.scale_array() + scale.shift_array()


def _check_channels(latent: npt.NDArray[np.float64], scale: LatentScale) -> None:
    if latent.ndim != 4 or latent.shape[0] != len(scale.shift):
        got = tuple(int(v) for v in latent.shape)
        expected = f"({len(scale.shift)}, frames, height, width)"
        raise ScaleError(f"{scale.name} latent must be {expected}, got {got}")
