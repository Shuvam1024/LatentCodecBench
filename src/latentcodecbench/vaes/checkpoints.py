"""Pinned VAE checkpoints.

Each entry is the Hugging Face repo, the git revision, and the
``vae/diffusion_pytorch_model.safetensors`` file that was on that revision
when the pin was taken (2026-10-19). Loading is :func:`latentcodecbench.vaes.load.load_vae`.
This module does not download anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VaeCheckpoint:
    """One VAE weight file and the license that file is published under."""

    name: str
    repo_id: str
    revision: str
    subfolder: str
    license_name: str
    license_url: str
    stored_dtype: str
    weight_bytes: int
    weight_sha256: str
    weight_commit: str


@dataclass(frozen=True)
class RelatedVaeFile:
    """A VAE file this benchmark does not load, recorded because it matches one it does."""

    repo_id: str
    revision: str
    weight_sha256: str


WAN2_1 = VaeCheckpoint(
    name="wan2_1",
    repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    revision="0fad780a534b6463e45facd96134c9f345acfa5b",
    subfolder="vae",
    license_name="Apache-2.0",
    license_url=(
        "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/blob/"
        "0fad780a534b6463e45facd96134c9f345acfa5b/README.md"
    ),
    stored_dtype="float32",
    weight_bytes=507_591_892,
    weight_sha256="d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
    weight_commit="5af5c01aff2e88e1a9d47bcd3dbba93822597f8e",
)
WAN2_2 = VaeCheckpoint(
    name="wan2_2",
    repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    revision="b8fff7315c768468a5333511427288870b2e9635",
    subfolder="vae",
    license_name="Apache-2.0",
    license_url=(
        "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/"
        "b8fff7315c768468a5333511427288870b2e9635/README.md"
    ),
    stored_dtype="float32",
    weight_bytes=2_818_777_808,
    weight_sha256="62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b",
    weight_commit="4c6ca6c2ded5c79550a3ca25555efc561112891a",
)
LTX_VIDEO = VaeCheckpoint(
    name="ltx",
    repo_id="Lightricks/LTX-Video",
    revision="8984fa25007f376c1a299016d0957a37a2f797bb",
    subfolder="vae",
    license_name="RAIL-M",
    license_url=(
        "https://huggingface.co/Lightricks/LTX-Video/blob/"
        "8984fa25007f376c1a299016d0957a37a2f797bb/ltx-video-2b-v0.9.license.txt"
    ),
    stored_dtype="float32",
    weight_bytes=1_676_798_532,
    weight_sha256="265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010",
    weight_commit="05be24f065a268e3b7881b63a114dbf52ce68fd1",
)
HUNYUAN = VaeCheckpoint(
    name="hunyuan",
    repo_id="hunyuanvideo-community/HunyuanVideo",
    revision="e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
    subfolder="vae",
    license_name="Tencent Hunyuan Community License Agreement",
    license_url=(
        "https://huggingface.co/tencent/HunyuanVideo/blob/"
        "6204ad6aea1a77ff5aba337c88278bb9500eb37d/LICENSE"
    ),
    stored_dtype="float32",
    weight_bytes=985_943_868,
    weight_sha256="7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883",
    weight_commit="bb5ecc0b6f0df267dd70ae80b0562301fc23096d",
)

CHECKPOINTS: dict[str, VaeCheckpoint] = {
    pin.name: pin for pin in (WAN2_1, WAN2_2, LTX_VIDEO, HUNYUAN)
}

# Lightest weights first, then the wide Wan2.2 decoder. The smoke uses this
# order so a 16 GB card measures the small models before the large one.
SMOKE_ORDER: tuple[str, ...] = ("wan2_1", "hunyuan", "ltx", "wan2_2")

# Wan2.2-T2V-A14B publishes the Wan2.1 VAE. Same file bytes, different
# ``_diffusers_version`` in vae/config.json. The benchmark loads Wan2.1.
WAN2_2_A14B_VAE = RelatedVaeFile(
    repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
    weight_sha256=WAN2_1.weight_sha256,
)
