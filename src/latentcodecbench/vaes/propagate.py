"""Turn a loaded video VAE into the encode/decode pair the propagation sweep calls.

Pixels follow the reconstruction sweep: y4m to RGB in ``[-1, 1]``, then
``(1, 3, frames, height, width)``. The encoded array is the posterior
mean on CPU. Decode returns luma only, because the propagation metrics
score luma.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from latentcodecbench.vaes.load import load_vae
from latentcodecbench.vaes.recon import _tensor_to_unit_rgb as tensor_to_unit_rgb
from latentcodecbench.vaes.recon import _unit_rgb_to_tensor as unit_rgb_to_tensor
from latentcodecbench.vaes.recon import (
    apply_sweep_tiling,
    unit_rgb_to_planes,
    y4m_to_unit_rgb,
)
from latentcodecbench.vaes.wrapper import VideoVae, _torch


class VideoVaeSession:
    """One VAE, already on the device the sweep will use."""

    def __init__(self, vae: VideoVae) -> None:
        torch = _torch()
        parameter = next(vae.module.parameters())
        self.vae = vae
        self._torch = torch
        self._dtype = parameter.dtype
        self._device = parameter.device

    def encode(self, y4m: Path) -> tuple[npt.NDArray[np.float32], float]:
        """Posterior mean, shape ``(1, channels, frames, height, width)``, and seconds."""
        _info, rgb = y4m_to_unit_rgb(y4m)
        video = unit_rgb_to_tensor(rgb, self._torch, dtype=self._dtype, device=self._device)
        _sync(self._torch)
        started = time.perf_counter()
        latents = self.vae.encode(video)
        _sync(self._torch)
        elapsed = time.perf_counter() - started
        array: npt.NDArray[np.float32] = latents.detach().float().cpu().numpy()
        return array, elapsed

    def decode_luma(self, latent: npt.NDArray[np.generic]) -> tuple[npt.NDArray[np.uint8], float]:
        """Luma of the decoded clip, shape ``(frames, height, width)``, and seconds."""
        tensor = self._torch.from_numpy(np.ascontiguousarray(latent)).to(
            device=self._device, dtype=self._dtype
        )
        _sync(self._torch)
        started = time.perf_counter()
        decoded = self.vae.decode(tensor)
        _sync(self._torch)
        elapsed = time.perf_counter() - started
        luma, _cb, _cr = unit_rgb_to_planes(tensor_to_unit_rgb(decoded))
        return luma, elapsed


def open_propagation_vae(name: str, *, dtype: Any, device: str) -> VideoVaeSession:
    """Load ``name`` and turn Hunyuan spatial tiling on before the first forward."""
    vae = load_vae(name, dtype=dtype, device=device)
    apply_sweep_tiling(name, vae.module)
    return VideoVaeSession(vae)


def _sync(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize()
