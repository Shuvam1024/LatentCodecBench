"""Random-init VAEs with the real compression factors and tiny widths.

The latent shape of these models is fixed by the channel count and the
downsample schedule, not by the width of the residual blocks. CI builds
these and never calls ``from_pretrained``.
"""

from __future__ import annotations

import importlib
from typing import Any

from latentcodecbench.vaes.spec import GEOMETRIES, VaeGeometry
from latentcodecbench.vaes.wrapper import VaeUnavailable, VideoVae


def _diffusers() -> Any:
    try:
        return importlib.import_module("diffusers")
    except ImportError as exc:
        raise VaeUnavailable("video VAEs need the vae extra (torch and diffusers)") from exc


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise VaeUnavailable("video VAEs need the vae extra (torch and diffusers)") from exc


def build_tiny(name: str, *, seed: int = 0) -> VideoVae:
    """Construct ``name`` with random weights on CPU.

    ``name`` is ``wan2_1``, ``wan2_2``, ``ltx``, or ``hunyuan``.
    """
    geometry = GEOMETRIES[name]
    torch = _torch()
    diffusers = _diffusers()
    torch.manual_seed(seed)
    cls = getattr(diffusers, geometry.diffusers_class)
    module = cls(**_tiny_kwargs(geometry))
    return VideoVae(geometry, module)


def _tiny_kwargs(geometry: VaeGeometry) -> dict[str, Any]:
    if geometry.name == "wan2_1":
        return {
            "base_dim": 4,
            "z_dim": geometry.latent_channels,
            "dim_mult": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "temperal_downsample": [False, True, True],
            "scale_factor_temporal": geometry.temporal_factor,
            "scale_factor_spatial": geometry.spatial_factor,
        }
    if geometry.name == "wan2_2":
        # patch_size 2 folds RGB into 12 channels, then the 8x conv stack
        # makes the published 16x spatial factor. The residual down block is
        # a weight-layout detail; the tiny net uses the plain downsample path.
        return {
            "base_dim": 4,
            "decoder_base_dim": 4,
            "z_dim": geometry.latent_channels,
            "dim_mult": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "temperal_downsample": [False, True, True],
            "in_channels": 12,
            "out_channels": 12,
            "patch_size": 2,
            "scale_factor_temporal": geometry.temporal_factor,
            "scale_factor_spatial": geometry.spatial_factor,
            "is_residual": False,
        }
    if geometry.name == "ltx":
        return {
            "latent_channels": geometry.latent_channels,
            "block_out_channels": (4, 4, 4, 4),
            "decoder_block_out_channels": (4, 4, 4, 4),
            "layers_per_block": (1, 1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, False),
            "decoder_spatio_temporal_scaling": (True, True, True, False),
            "decoder_inject_noise": (False, False, False, False, False),
            "downsample_type": ("conv", "conv", "conv", "conv"),
            "upsample_residual": (False, False, False, False),
            "upsample_factor": (1, 1, 1, 1),
            "patch_size": 4,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
        }
    if geometry.name == "hunyuan":
        return {
            "latent_channels": geometry.latent_channels,
            "block_out_channels": (4, 4, 4, 4),
            "layers_per_block": 1,
            "norm_num_groups": 1,
            "spatial_compression_ratio": geometry.spatial_factor,
            "temporal_compression_ratio": geometry.temporal_factor,
            "mid_block_add_attention": False,
        }
    raise KeyError(geometry.name)
