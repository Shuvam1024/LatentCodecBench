"""Shape contract for the four video VAEs.

Geometry tests run with no torch. The forward tests build a tiny random-init
network and require the ``vae`` extra. They fail if a checkpoint download is
attempted.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import pytest

from latentcodecbench.data.canonical import CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH
from latentcodecbench.vaes import (
    DIFFUSERS_PIN,
    GEOMETRIES,
    HUNYUAN,
    LTX_VIDEO,
    WAN2_1,
    WAN2_2,
    latent_shape,
)
from latentcodecbench.vaes.tiny import build_tiny

# (channels, latent_frames, height, width) for one 832x480x33 clip.
CANONICAL_LATENTS = {
    "wan2_1": (16, 9, 60, 104),
    "hunyuan": (16, 9, 60, 104),
    "wan2_2": (48, 9, 30, 52),
    "ltx": (128, 5, 15, 26),
}


def _require_vae() -> Any:
    """Import torch, or skip when the extra is absent.

    CI sets ``LATENTCODECBENCH_REQUIRE_VAE=1`` so a missing extra fails the job
    instead of skipping the shape contract.
    """
    try:
        return importlib.import_module("torch")
    except ImportError:
        if os.environ.get("LATENTCODECBENCH_REQUIRE_VAE") == "1":
            pytest.fail("CI requires the vae extra; torch is not installed")
        pytest.importorskip("torch")
    raise AssertionError("unreachable")


def test_canonical_latent_shapes() -> None:
    clips = {
        "wan2_1": WAN2_1,
        "wan2_2": WAN2_2,
        "ltx": LTX_VIDEO,
        "hunyuan": HUNYUAN,
    }
    for name, geometry in clips.items():
        assert (
            latent_shape(geometry, CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH)
            == (CANONICAL_LATENTS[name])
        )


def test_illegal_frame_count_and_resolution() -> None:
    with pytest.raises(ValueError, match="4k\\+1"):
        latent_shape(WAN2_1, 32, CANONICAL_HEIGHT, CANONICAL_WIDTH)
    with pytest.raises(ValueError, match="8k\\+1"):
        latent_shape(LTX_VIDEO, 32, CANONICAL_HEIGHT, CANONICAL_WIDTH)
    # 816 is divisible by 16 and not by 32, so LTX rejects it and Wan2.2 would accept it.
    with pytest.raises(ValueError, match="divisible by 32"):
        latent_shape(LTX_VIDEO, 9, 480, 816)


def test_diffusers_pin_matches_the_installed_extra() -> None:
    if os.environ.get("LATENTCODECBENCH_REQUIRE_VAE") != "1":
        pytest.importorskip("diffusers")
    from importlib.metadata import version

    assert version("diffusers") == DIFFUSERS_PIN


def test_tiny_forward_matches_canonical_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _require_vae()
    _forbid_checkpoint_fetch(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    video = torch.zeros(1, 3, CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH)
    for name, expected in CANONICAL_LATENTS.items():
        vae = build_tiny(name)
        latents = vae.encode(video)
        assert tuple(int(v) for v in latents.shape) == (1, *expected)
        module = vae.module
        assert int(module.spatial_compression_ratio) == GEOMETRIES[name].spatial_factor
        temporal = getattr(module, "temporal_compression_ratio", None)
        if temporal is None:
            temporal = module.config.scale_factor_temporal
        assert int(temporal) == GEOMETRIES[name].temporal_factor


def test_tiny_decode_restores_a_small_legal_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _require_vae()
    _forbid_checkpoint_fetch(monkeypatch)
    # 32x32x9 is legal for every factor in this set: 9 = 4*2+1 = 8*1+1,
    # and 32 is divisible by 8, 16, and 32.
    video = torch.zeros(1, 3, 9, 32, 32)
    for name in CANONICAL_LATENTS:
        vae = build_tiny(name)
        decoded = vae.decode(vae.encode(video))
        assert tuple(int(v) for v in decoded.shape) == (1, 3, 9, 32, 32)


def _forbid_checkpoint_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("checkpoint fetch is not allowed in CPU tests")

    hub = importlib.import_module("huggingface_hub")
    file_download = importlib.import_module("huggingface_hub.file_download")
    monkeypatch.setattr(hub, "hf_hub_download", boom)
    monkeypatch.setattr(file_download, "hf_hub_download", boom)
    monkeypatch.setattr(hub, "snapshot_download", boom)
    diffusers = importlib.import_module("diffusers")
    monkeypatch.setattr(diffusers.ModelMixin, "from_pretrained", boom)
