"""Encode and decode through one video VAE.

The module behind a :class:`VideoVae` is already constructed. Loading a
checkpoint is a later step. Pixels are ``(batch, 3, frames, height, width)``
in ``[-1, 1]``. Latents are ``(batch, channels, latent_frames, height, width)``,
and the encoded tensor is the posterior mean.
"""

from __future__ import annotations

import importlib
from typing import Any

from latentcodecbench.vaes.spec import VaeGeometry, latent_shape


class VaeUnavailable(RuntimeError):
    """Torch or Diffusers is not installed."""


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise VaeUnavailable("video VAEs need the vae extra (torch and diffusers)") from exc


class VideoVae:
    """One constructed VAE plus the geometry that says what shape is legal."""

    def __init__(self, geometry: VaeGeometry, module: Any) -> None:
        self.geometry = geometry
        self.module = module
        module.eval()

    def encode(self, video: Any) -> Any:
        """Posterior mean. ``video`` is NCHW with a time axis in the middle, N,C,T,H,W."""
        torch = _torch()
        if video.ndim != 5 or video.shape[1] != 3:
            got = tuple(video.shape)
            raise ValueError(f"video must be (batch, 3, frames, height, width), got {got}")
        expected = latent_shape(
            self.geometry,
            frames=int(video.shape[2]),
            height=int(video.shape[3]),
            width=int(video.shape[4]),
        )
        with torch.no_grad():
            encoded = self.module.encode(video)
            posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
            latents = posterior.mode()
        got = tuple(int(v) for v in latents.shape[1:])
        if got != expected:
            raise RuntimeError(f"{self.geometry.name} latent shape {got} != {expected}")
        return latents

    def decode(self, latents: Any) -> Any:
        """Pixels in ``[-1, 1]``, shape ``(batch, 3, frames, height, width)``."""
        torch = _torch()
        with torch.no_grad():
            decoded = self.module.decode(latents)
        sample = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return sample
