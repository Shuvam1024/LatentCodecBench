"""Reconstruction sweep: resume, empty rate columns, tiling, chunk timing.

The forward tests build a tiny random-init network. They do not download
checkpoints and they do not need a GPU.
"""

from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from latentcodecbench.analysis.table import read_results
from latentcodecbench.data.manifest import load_manifest
from latentcodecbench.vaes import GEOMETRIES
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER
from latentcodecbench.vaes.recon import (
    FULL_SWEEP_ROWS,
    ClipReconstruction,
    ReconstructionError,
    apply_sweep_tiling,
    load_recon_config,
    merge_reconstruction,
    reconstruct_tensor,
    run_reconstruction,
)
from latentcodecbench.vaes.smoke import tiling_flags
from latentcodecbench.vaes.spec import WAN2_1, latent_shape
from latentcodecbench.vaes.wrapper import VideoVae

_MANIFEST = """\
version: 1
canonical:
  width: 832
  height: 480
  num_frames: 33
  pix_fmt: yuv420p
clips:
{clips}
"""

_CLIP = """\
  - id: {clip_id}
    source_url: file:///tmp/{clip_id}.y4m
    sha256: {sha}
    license: test
    license_url: https://example.com/license
    native_width: 832
    native_height: 480
    native_fps_num: 24
    native_fps_den: 1
    category: test
"""


class _Module:
    def __init__(self) -> None:
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

    def eval(self) -> _Module:
        return self

    def enable_tiling(self) -> None:
        self.use_tiling = True


def _require_vae() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        if os.environ.get("LATENTCODECBENCH_REQUIRE_VAE") == "1":
            pytest.fail("CI requires the vae extra; torch is not installed")
        pytest.importorskip("torch")
    raise AssertionError("unreachable")


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


def _repo_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "vae-recon.yaml"


def _write_manifest(directory: Path, clip_ids: list[str]) -> Path:
    clips = "\n".join(_CLIP.format(clip_id=clip_id, sha="ab" * 32) for clip_id in clip_ids)
    path = directory / "clips.yaml"
    path.write_text(_MANIFEST.format(clips=clips), encoding="utf-8")
    clip_dir = directory / "clips"
    clip_dir.mkdir()
    for clip_id in clip_ids:
        (clip_dir / f"{clip_id}.y4m").write_bytes(b"")
    return path


def _loader(name: str) -> VideoVae:
    module = _Module()
    if name == "hunyuan":
        module.use_framewise_encoding = True
        module.use_framewise_decoding = True
    return VideoVae(GEOMETRIES[name], module)


def _measured(vae: VideoVae, *, psnr_y: float = 40.0) -> ClipReconstruction:
    flags = tiling_flags(vae.module)
    return ClipReconstruction(
        encode_seconds=0.2,
        decode_seconds=0.3,
        peak_vram_bytes=1024,
        spatial_tiling=flags["spatial_tiling"],
        framewise_encoding=flags["framewise_encoding"],
        framewise_decoding=flags["framewise_decoding"],
        retried_with_tiling=False,
        chunk_count=3,
        per_chunk_seconds=0.01,
        latent_shape="16x9x60x104",
        psnr_y=psnr_y,
        psnr_rgb=39.0,
        ssim=0.95,
        vmaf=90.0,
        flicker=0.1,
        width=832,
        height=480,
        frames=33,
        fps_num=24,
        fps_den=1,
    )


def test_sweep_notebook_states_the_colab_contract() -> None:
    text = (Path(__file__).resolve().parents[1] / "notebooks" / "vae_sweep.ipynb").read_text(
        encoding="utf-8"
    )
    for needle in (
        "torchao",
        "latentcodecbench-*.whl",
        "/content/results/vae-recon.csv",
        "manifests/clips.yaml",
        "bfloat16",
        "float16",
        "scripts/merge-vae-recon.py",
        "68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269",
    ):
        assert needle in text


def test_packaged_config_matches_the_repo_file() -> None:
    import latentcodecbench.vaes as vaes

    packaged = Path(vaes.__file__).resolve().parent / "recon_config.yaml"
    assert packaged.read_bytes() == _repo_config().read_bytes()
    assert load_recon_config(packaged) == SMOKE_ORDER


def test_clip_set_is_136_reconstructions() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "manifests" / "clips.yaml")
    assert len(manifest.clips) == 34
    assert len(SMOKE_ORDER) == 4
    assert len(manifest.clips) * len(SMOKE_ORDER) == FULL_SWEEP_ROWS


def test_sweep_resumes_and_leaves_rate_columns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "d" * 40)
    manifest = _write_manifest(tmp_path, ["beta", "alpha"])
    out = tmp_path / "vae-recon.csv"
    calls: list[tuple[str, str]] = []

    def runner(vae: VideoVae, source: Path, work_dir: Path) -> ClipReconstruction:
        if calls:
            lines = out.read_text(encoding="utf-8").splitlines()
            assert len(lines) == len(calls) + 1
        else:
            assert not out.is_file()
        calls.append((source.stem, vae.geometry.name))
        return _measured(vae)

    moment = datetime(2026, 10, 19, 23, 0, tzinfo=UTC)
    rows = run_reconstruction(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.14.1",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        now=moment,
        runner=runner,
    )
    assert len(rows) == 8
    assert len(calls) == 8
    assert rows[0].run_id.startswith("20261019T230000Z-dddddddddddd-")
    assert rows[0].crf is None
    assert rows[0].bits is None
    assert rows[0].bpp is None
    assert rows[0].kbps is None
    assert rows[0].lpips is None
    assert rows[0].preset == "reconstruction"
    hunyuan = [row for row in rows if row.codec == "hunyuan"]
    assert hunyuan
    assert all(row.spatial_tiling is True for row in hunyuan)
    assert all(row.framewise_encoding is True for row in hunyuan)
    others = [row for row in rows if row.codec != "hunyuan"]
    assert all(row.spatial_tiling is False for row in others)
    assert [(row.clip_id, row.codec) for row in rows] == [
        ("alpha", "hunyuan"),
        ("alpha", "ltx"),
        ("alpha", "wan2_1"),
        ("alpha", "wan2_2"),
        ("beta", "hunyuan"),
        ("beta", "ltx"),
        ("beta", "wan2_1"),
        ("beta", "wan2_2"),
    ]
    loaded = read_results(out)
    assert loaded[0].peak_vram_bytes == 1024
    assert loaded[0].torch_version == "2.14.1"
    assert loaded[0].runtime_dtype == "bfloat16"
    assert "peak_vram_bytes" in out.read_text(encoding="utf-8").splitlines()[0]

    calls.clear()
    again = run_reconstruction(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.14.1",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        now=datetime(2026, 1, 1, tzinfo=UTC),
        runner=runner,
    )
    assert calls == []
    assert again[0].run_id == rows[0].run_id


def test_a_failed_clip_is_skipped_and_a_different_config_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "e" * 40)
    manifest = _write_manifest(tmp_path, ["bad", "good"])
    out = tmp_path / "vae-recon.csv"

    def runner(vae: VideoVae, source: Path, _work: Path) -> ClipReconstruction:
        if source.stem == "bad":
            raise RuntimeError("CUDA out of memory")
        return _measured(vae)

    rows = run_reconstruction(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.14.1",
        diffusers_version="0.41.0",
        runtime_dtype="float16",
        gpu_name="Tesla T4",
        runner=runner,
    )
    assert {row.clip_id for row in rows} == {"good"}
    other = tmp_path / "other.yaml"
    other.write_text(_repo_config().read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ReconstructionError, match="different config"):
        run_reconstruction(
            manifest_path=manifest,
            config_path=other,
            clip_dir=tmp_path / "clips",
            out=out,
            work_dir=tmp_path / "work",
            loader=_loader,
            torch_version="2.14.1",
            diffusers_version="0.41.0",
            runtime_dtype="float16",
            gpu_name="Tesla T4",
            runner=runner,
        )


def test_merge_unions_rows_and_refuses_a_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "f" * 40)
    manifest = _write_manifest(tmp_path, ["alpha"])
    first = tmp_path / "first.csv"
    second_dir = tmp_path / "more"
    second_dir.mkdir()
    second_manifest = _write_manifest(second_dir, ["beta"])
    second = second_dir / "second.csv"

    def runner(vae: VideoVae, _source: Path, _work: Path) -> ClipReconstruction:
        return _measured(vae)

    run_reconstruction(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=first,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.14.1",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        runner=runner,
    )
    run_reconstruction(
        manifest_path=second_manifest,
        config_path=_repo_config(),
        clip_dir=second_dir / "clips",
        out=second,
        work_dir=second_dir / "work",
        loader=_loader,
        torch_version="2.14.1",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        runner=runner,
    )
    dest = tmp_path / "docs" / "vae-recon.csv"
    merged = merge_reconstruction(first, dest, config_path=_repo_config())
    merged = merge_reconstruction(second, dest, config_path=_repo_config())
    assert [(row.clip_id, row.codec) for row in merged] == [
        ("alpha", name) for name in ("hunyuan", "ltx", "wan2_1", "wan2_2")
    ] + [("beta", name) for name in ("hunyuan", "ltx", "wan2_1", "wan2_2")]

    changed = tmp_path / "changed.csv"
    text = first.read_text(encoding="utf-8").replace("40.0", "41.0", 1)
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ReconstructionError, match="different row"):
        merge_reconstruction(changed, dest, config_path=_repo_config())


def test_merge_rejects_hunyuan_without_spatial_tiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENTCODECBENCH_GIT_SHA", "a" * 40)
    manifest = _write_manifest(tmp_path, ["alpha"])
    out = tmp_path / "vae-recon.csv"

    def runner(vae: VideoVae, _source: Path, _work: Path) -> ClipReconstruction:
        measured = _measured(vae)
        if vae.geometry.name != "hunyuan":
            return measured
        return ClipReconstruction(
            encode_seconds=measured.encode_seconds,
            decode_seconds=measured.decode_seconds,
            peak_vram_bytes=measured.peak_vram_bytes,
            spatial_tiling=False,
            framewise_encoding=measured.framewise_encoding,
            framewise_decoding=measured.framewise_decoding,
            retried_with_tiling=False,
            chunk_count=measured.chunk_count,
            per_chunk_seconds=measured.per_chunk_seconds,
            latent_shape=measured.latent_shape,
            psnr_y=measured.psnr_y,
            psnr_rgb=measured.psnr_rgb,
            ssim=measured.ssim,
            vmaf=measured.vmaf,
            flicker=measured.flicker,
            width=measured.width,
            height=measured.height,
            frames=measured.frames,
            fps_num=measured.fps_num,
            fps_den=measured.fps_den,
        )

    run_reconstruction(
        manifest_path=manifest,
        config_path=_repo_config(),
        clip_dir=tmp_path / "clips",
        out=out,
        work_dir=tmp_path / "work",
        loader=_loader,
        torch_version="2.14.1",
        diffusers_version="0.41.0",
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        only={"alpha"},
        runner=runner,
    )
    with pytest.raises(ReconstructionError, match="spatial tiling"):
        merge_reconstruction(out, tmp_path / "dest.csv", config_path=_repo_config())


def test_hunyuan_tiling_is_on_before_the_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_vae()
    _forbid_checkpoint_fetch(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from latentcodecbench.vaes.tiny import build_tiny

    hunyuan = build_tiny("hunyuan")
    assert hunyuan.module.use_tiling is False
    assert hunyuan.module.use_framewise_encoding is True
    apply_sweep_tiling("hunyuan", hunyuan.module)
    assert hunyuan.module.use_tiling is True
    assert hunyuan.module.use_framewise_encoding is True
    wan = build_tiny("wan2_1")
    apply_sweep_tiling("wan2_1", wan.module)
    assert bool(wan.module.use_tiling) is False


def test_wan_counts_one_encoder_call_per_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _require_vae()
    _forbid_checkpoint_fetch(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from latentcodecbench.vaes.tiny import build_tiny

    vae = build_tiny("wan2_1")
    video = torch.zeros(1, 3, 9, 32, 32)
    measured = reconstruct_tensor(vae, video)
    assert measured.chunk_count == 1 + (9 - 1) // 4
    assert measured.per_chunk_seconds is not None
    assert measured.per_chunk_seconds > 0
    assert measured.retried_with_tiling is False
    assert measured.spatial_tiling is False
    assert measured.latent_shape == "x".join(str(part) for part in latent_shape(WAN2_1, 9, 32, 32))
    assert tuple(int(v) for v in measured.pixels.shape) == (1, 3, 9, 32, 32)


def test_out_of_memory_retries_once_with_tiling(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _require_vae()
    _forbid_checkpoint_fetch(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from latentcodecbench.vaes.tiny import build_tiny

    vae = build_tiny("wan2_1")
    original = vae.encode
    calls = {"n": 0}

    def encode(video: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original(video)

    vae.encode = encode  # type: ignore[method-assign]
    measured = reconstruct_tensor(vae, torch.zeros(1, 3, 9, 32, 32))
    assert calls["n"] == 2
    assert measured.retried_with_tiling is True
    assert measured.spatial_tiling is True

    def always_oom(_video: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory")

    vae.encode = always_oom  # type: ignore[assignment]
    with pytest.raises(torch.OutOfMemoryError):
        reconstruct_tensor(vae, torch.zeros(1, 3, 9, 32, 32))
