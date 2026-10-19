"""Pinned VAE checkpoints and the offline load contract.

Nothing in this file downloads a weight. The forward uses the tiny
random-init classes. ``load_vae`` is tested against a patched
``from_pretrained``.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from latentcodecbench.vaes import GEOMETRIES
from latentcodecbench.vaes.checkpoints import (
    CHECKPOINTS,
    SMOKE_ORDER,
    WAN2_1,
    WAN2_2_A14B_VAE,
)
from latentcodecbench.vaes.load import load_vae
from latentcodecbench.vaes.smoke import (
    PSNR_MAX_VALUE,
    SMOKE_FIELDS,
    load_failure_row,
    measure_reconstruction,
    runtime_dtype,
    runtime_dtype_name,
    write_smoke,
)
from latentcodecbench.vaes.tiny import build_tiny


def _require_vae() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        if os.environ.get("LATENTCODECBENCH_REQUIRE_VAE") == "1":
            pytest.fail("CI requires the vae extra; torch is not installed")
        pytest.importorskip("torch")
    raise AssertionError("unreachable")


def test_checkpoint_pins_match_the_geometry_repos() -> None:
    assert set(CHECKPOINTS) == set(GEOMETRIES)
    assert tuple(SMOKE_ORDER) == tuple(sorted(CHECKPOINTS, key=SMOKE_ORDER.index))
    for name, pin in CHECKPOINTS.items():
        assert pin.name == name
        assert pin.repo_id == GEOMETRIES[name].repo_id
        assert pin.subfolder == "vae"
        assert pin.stored_dtype == "float32"
        assert len(pin.revision) == 40
        assert len(pin.weight_sha256) == 64
        assert len(pin.weight_commit) == 40
        int(pin.revision, 16)
        int(pin.weight_sha256, 16)
        int(pin.weight_commit, 16)
        assert pin.weight_bytes > 0
        assert pin.license_name
        assert pin.license_url.startswith("https://huggingface.co/")


def test_wan22_a14b_vae_file_matches_wan21() -> None:
    assert WAN2_2_A14B_VAE.weight_sha256 == WAN2_1.weight_sha256
    assert WAN2_2_A14B_VAE.repo_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert len(WAN2_2_A14B_VAE.revision) == 40


def test_unknown_name_raises_before_any_download() -> None:
    with pytest.raises(KeyError, match="nope"):
        load_vae("nope")


def test_load_vae_requests_the_vae_subfolder_only(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _require_vae()
    captured: dict[str, Any] = {}

    class _Module:
        def eval(self) -> None:
            return None

        def to(self, device: object) -> _Module:
            captured["device"] = device
            return self

    def from_pretrained(repo: str, **kwargs: object) -> _Module:
        captured["repo"] = repo
        captured["kwargs"] = kwargs
        return _Module()

    diffusers = importlib.import_module("diffusers")
    monkeypatch.setattr(diffusers.AutoencoderKLWan, "from_pretrained", from_pretrained)
    vae = load_vae("wan2_1", dtype=torch.float32, device="cpu")
    assert vae.geometry.name == "wan2_1"
    assert captured["repo"] == CHECKPOINTS["wan2_1"].repo_id
    assert captured["device"] == "cpu"
    assert captured["kwargs"] == {
        "subfolder": "vae",
        "revision": CHECKPOINTS["wan2_1"].revision,
        "torch_dtype": torch.float32,
    }


def test_runtime_dtype_follows_compute_capability() -> None:
    class _Cuda:
        def __init__(self, available: bool, major: int) -> None:
            self._available = available
            self._major = major

        def is_available(self) -> bool:
            return self._available

        def get_device_capability(self) -> tuple[int, int]:
            return (self._major, 0)

    class _Torch:
        def __init__(self, cuda: _Cuda) -> None:
            self.cuda = cuda
            self.float16 = "float16"
            self.bfloat16 = "bfloat16"
            self.float32 = "float32"

    t4 = _Torch(_Cuda(True, 7))
    ampere = _Torch(_Cuda(True, 8))
    cpu = _Torch(_Cuda(False, 0))
    assert runtime_dtype_name(t4) == "float16"
    assert runtime_dtype(t4) == "float16"
    assert runtime_dtype_name(ampere) == "bfloat16"
    assert runtime_dtype_name(cpu) == "float32"


def test_tiny_roundtrip_records_shape_and_default_tiling() -> None:
    torch = _require_vae()
    video = torch.zeros(1, 3, 9, 32, 32)
    wan = measure_reconstruction(build_tiny("wan2_1"), video, runtime_dtype="float32")
    assert wan["subfolder"] == "vae"
    assert wan["revision"] == CHECKPOINTS["wan2_1"].revision
    assert wan["oom"] is False
    assert wan["error"] is None
    assert wan["retried_with_tiling"] is False
    assert wan["spatial_tiling"] is False
    assert wan["framewise_encoding"] is False
    assert wan["latent_shape"] == [16, 3, 4, 4]
    assert wan["psnr_max_value"] == PSNR_MAX_VALUE
    assert wan["encode_seconds"] >= 0
    hunyuan = measure_reconstruction(build_tiny("hunyuan"), video, runtime_dtype="float32")
    assert hunyuan["latent_shape"] == [16, 3, 4, 4]
    assert hunyuan["framewise_encoding"] is True
    assert hunyuan["framewise_decoding"] is True
    assert hunyuan["spatial_tiling"] is False


def test_out_of_memory_retries_once_with_tiling() -> None:
    torch = _require_vae()
    vae = build_tiny("wan2_1")
    calls = {"n": 0}
    real_encode = vae.encode

    def flaky(video: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        return real_encode(video)

    vae.encode = flaky  # type: ignore[method-assign]
    video = torch.zeros(1, 3, 9, 32, 32)
    row = measure_reconstruction(vae, video, runtime_dtype="float16")
    assert calls["n"] == 2
    assert row["retried_with_tiling"] is True
    assert row["spatial_tiling"] is True
    assert row["oom"] is False
    assert row["latent_shape"] == [16, 3, 4, 4]


def test_second_out_of_memory_is_recorded() -> None:
    _require_vae()
    vae = build_tiny("ltx")

    def always_oom(_video: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    vae.encode = always_oom  # type: ignore[method-assign,assignment]
    row = measure_reconstruction(vae, object(), runtime_dtype="float16")
    assert row["oom"] is True
    assert row["retried_with_tiling"] is True
    assert row["framewise_decoding"] is True
    assert row["psnr_db"] is None
    assert row["error"] is not None
    assert "out of memory" in row["error"]


def test_load_failure_row_keeps_the_pin() -> None:
    row = load_failure_row("wan2_2", "float16", RuntimeError("disk full"))
    assert row["repo_id"] == CHECKPOINTS["wan2_2"].repo_id
    assert row["subfolder"] == "vae"
    assert row["oom"] is False
    assert row["error"] == "RuntimeError: disk full"


def test_write_smoke_json_and_csv(tmp_path: Path) -> None:
    row = load_failure_row("wan2_1", "float16", MemoryError("out of memory"))
    clip = {"frames": 33, "height": 480, "width": 832, "seed": 0}
    json_path, csv_path = write_smoke(tmp_path / "vae-smoke.json", [row], clip=clip)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["clip"] == clip
    assert document["rows"][0]["revision"] == CHECKPOINTS["wan2_1"].revision
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(SMOKE_FIELDS)
    assert "out of memory" in csv_path.read_text(encoding="utf-8")
