"""Geometry of the four video VAEs, with no weights and no torch.

The numbers are the ones Diffusers 0.41.0 reads from each VAE config.
Wan2.1 and HunyuanVideo compress by 4 in time and 8 in space, with 16
latent channels. The Wan2.2 TI2V-5B VAE compresses by 4 and 16, with 48
channels. LTX-Video compresses by 8 and 32, with 128 channels.

A causal encoder keeps the first frame by itself and then groups of
``temporal_factor`` frames, so a legal length is ``temporal_factor * k + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Released 2026-10-06. The class files checked for this pin are
# autoencoder_kl_wan.py, autoencoder_kl_ltx.py, and
# autoencoder_kl_hunyuan_video.py on tag v0.41.0.
DIFFUSERS_PIN = "0.41.0"


@dataclass(frozen=True)
class VaeGeometry:
    """Compression factors for one VAE family.

    ``repo_id`` names the Diffusers repository the weights will come from.
    This module does not download it.
    """

    name: str
    diffusers_class: str
    repo_id: str
    latent_channels: int
    spatial_factor: int
    temporal_factor: int


WAN2_1 = VaeGeometry(
    name="wan2_1",
    diffusers_class="AutoencoderKLWan",
    repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    latent_channels=16,
    spatial_factor=8,
    temporal_factor=4,
)
WAN2_2 = VaeGeometry(
    name="wan2_2",
    diffusers_class="AutoencoderKLWan",
    repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    latent_channels=48,
    spatial_factor=16,
    temporal_factor=4,
)
LTX_VIDEO = VaeGeometry(
    name="ltx",
    diffusers_class="AutoencoderKLLTXVideo",
    repo_id="Lightricks/LTX-Video",
    latent_channels=128,
    spatial_factor=32,
    temporal_factor=8,
)
HUNYUAN = VaeGeometry(
    name="hunyuan",
    diffusers_class="AutoencoderKLHunyuanVideo",
    repo_id="hunyuanvideo-community/HunyuanVideo",
    latent_channels=16,
    spatial_factor=8,
    temporal_factor=4,
)

GEOMETRIES: dict[str, VaeGeometry] = {
    spec.name: spec for spec in (WAN2_1, WAN2_2, LTX_VIDEO, HUNYUAN)
}


def latent_shape(
    geometry: VaeGeometry,
    frames: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Latent shape ``(channels, frames, height, width)`` for one clip.

    Batch is not included. The frame count is ``(frames - 1) / temporal + 1``.
    """
    if frames < 1 or (frames - 1) % geometry.temporal_factor != 0:
        raise ValueError(
            f"{geometry.name} wants {geometry.temporal_factor}k+1 frames, got {frames}"
        )
    if height % geometry.spatial_factor != 0 or width % geometry.spatial_factor != 0:
        raise ValueError(
            f"{geometry.name} wants height and width divisible by "
            f"{geometry.spatial_factor}, got {height}x{width}"
        )
    latent_frames = (frames - 1) // geometry.temporal_factor + 1
    return (
        geometry.latent_channels,
        latent_frames,
        height // geometry.spatial_factor,
        width // geometry.spatial_factor,
    )
