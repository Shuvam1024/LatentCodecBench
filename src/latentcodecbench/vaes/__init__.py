"""Video VAE wrappers.

Phase 2 treats Wan2.1, Wan2.2, LTX-Video, and HunyuanVideo as codecs.
:class:`~latentcodecbench.vaes.wrapper.VideoVae` exposes encode and decode.
Checkpoint pins live in :mod:`latentcodecbench.vaes.checkpoints`.
:func:`latentcodecbench.vaes.load.load_vae` reads one of them from
``subfolder="vae"``. Importing this package does not import torch and does
not download weights.

The library pin is :data:`~latentcodecbench.vaes.spec.DIFFUSERS_PIN`.
"""

from latentcodecbench.vaes.checkpoints import CHECKPOINTS, VaeCheckpoint
from latentcodecbench.vaes.load import load_vae
from latentcodecbench.vaes.spec import (
    DIFFUSERS_PIN,
    GEOMETRIES,
    HUNYUAN,
    LTX_VIDEO,
    WAN2_1,
    WAN2_2,
    VaeGeometry,
    latent_shape,
)
from latentcodecbench.vaes.wrapper import VaeUnavailable, VideoVae

__all__ = [
    "CHECKPOINTS",
    "DIFFUSERS_PIN",
    "GEOMETRIES",
    "HUNYUAN",
    "LTX_VIDEO",
    "WAN2_1",
    "WAN2_2",
    "VaeCheckpoint",
    "VaeGeometry",
    "VaeUnavailable",
    "VideoVae",
    "latent_shape",
    "load_vae",
]
