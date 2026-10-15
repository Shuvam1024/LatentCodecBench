"""Video VAE wrappers.

Phase 2 treats Wan2.1, Wan2.2, LTX-Video, and HunyuanVideo as codecs.
:class:`~latentcodecbench.vaes.wrapper.VideoVae` exposes encode, decode, and
the compression factors. Weights are not loaded here. Torch and Diffusers
stay optional: importing this package does not import them.

The pin is :data:`~latentcodecbench.vaes.spec.DIFFUSERS_PIN`.
"""

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
    "DIFFUSERS_PIN",
    "GEOMETRIES",
    "HUNYUAN",
    "LTX_VIDEO",
    "WAN2_1",
    "WAN2_2",
    "VaeGeometry",
    "VaeUnavailable",
    "VideoVae",
    "latent_shape",
]
