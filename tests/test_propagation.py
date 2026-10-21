"""Phase 3 CPU pieces: degraded sources, artifact metrics, and the local CSV."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from latentcodecbench.analysis.handoff import (
    _child as handoff_child,
)
from latentcodecbench.analysis.handoff import (
    directory_state,
    pull_resume,
    push_copy,
    stage_files,
)
from latentcodecbench.analysis.propagation import (
    format_duration,
    l4_runtime_markdown,
    propagation_gpu_seconds,
    read_propagation,
    run_propagation,
    study_variant_count,
)
from latentcodecbench.codecs.baseline import (
    BENCHMARK_PRESET,
    SVT_AV1_CRF,
    X264_CRF,
    X265_CRF,
)
from latentcodecbench.codecs.degrade import (
    PHASE1_LADDER,
    STUDY_LADDER,
    degrade_one,
    degrade_sources,
)
from latentcodecbench.data.preprocess import ffmpeg_binary
from latentcodecbench.data.y4m import read_y4m, write_y4m
from latentcodecbench.metrics.artifacts import (
    artifact_retention,
    blocking_score,
    latent_distance,
    propagation_quality,
    ringing_score,
)
from latentcodecbench.metrics.psnr import psnr


def test_study_ladder_is_three_phase1_points() -> None:
    assert STUDY_LADDER == {
        "libx264": (18, 30, 38),
        "libx265": (18, 30, 38),
        "libsvtav1": (28, 49, 63),
    }
    assert study_variant_count() == 9
    assert PHASE1_LADDER == {
        "libx264": X264_CRF,
        "libx265": X265_CRF,
        "libsvtav1": SVT_AV1_CRF,
    }
    for codec, crfs in STUDY_LADDER.items():
        assert len(crfs) == 3
        assert all(crf in PHASE1_LADDER[codec] for crf in crfs)
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "artifact-sources.yaml").read_text(encoding="utf-8"))
    assert raw["kind"] == "artifact-sources"
    for codec, crfs in STUDY_LADDER.items():
        block = raw["codecs"][codec]
        assert tuple(block["crf"]) == crfs
        assert str(block["preset"]) == BENCHMARK_PRESET[codec]


def test_latent_distance_locks_mse_and_cosine() -> None:
    clean = np.ones((2, 2), dtype=np.float64)
    same = latent_distance(clean, clean.copy())
    assert same.mse == 0.0
    assert same.relative_l2 == 0.0
    assert same.cosine == 1.0
    shifted = latent_distance(clean, clean + 3.0)
    assert shifted.mse == pytest.approx(9.0)
    assert shifted.relative_l2 == pytest.approx(3.0)
    assert shifted.cosine == pytest.approx(1.0)
    turned = latent_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert turned.cosine == pytest.approx(0.0)
    assert turned.mse == pytest.approx(1.0)
    zeros = np.zeros(4, dtype=np.float64)
    origin = latent_distance(zeros, zeros)
    assert origin.cosine == 1.0
    assert origin.relative_l2 == 0.0
    against = latent_distance(zeros, np.ones(4))
    assert against.cosine == 0.0
    assert against.relative_l2 == pytest.approx(2.0)
    with pytest.raises(ValueError, match="shape"):
        latent_distance(clean, np.ones(3))
    with pytest.raises(ValueError, match="at least one"):
        latent_distance(np.ones(0), np.ones(0))


def test_blocking_score_is_zero_on_a_ramp_and_high_on_a_block_edge() -> None:
    constant = np.full((16, 16), 40, dtype=np.uint8)
    assert blocking_score(constant) == pytest.approx(0.0)
    ramp = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    assert blocking_score(ramp) == pytest.approx(0.0)
    split = np.zeros((16, 16), dtype=np.uint8)
    split[:, 8:] = 255
    assert blocking_score(split) == pytest.approx(127.5)
    stacked = np.stack([split, split])
    assert blocking_score(stacked) == pytest.approx(127.5)
    with pytest.raises(ValueError, match="multiples of 8"):
        blocking_score(np.zeros((16, 15), dtype=np.uint8))


def test_ringing_is_zero_on_a_step_and_positive_on_an_overshoot() -> None:
    step = np.zeros((16, 32), dtype=np.uint8)
    step[:, 16:] = 128
    overshoot = step.copy()
    overshoot[:, 16] = 160
    assert ringing_score(step) == pytest.approx(0.0)
    assert ringing_score(overshoot) > 0.0
    assert ringing_score(np.full((16, 32), 30, dtype=np.uint8)) == pytest.approx(0.0)


def test_retention_is_zero_when_removed_and_one_when_kept() -> None:
    assert artifact_retention(0.0, 10.0, 0.0) == pytest.approx(0.0)
    assert artifact_retention(0.0, 10.0, 10.0) == pytest.approx(1.0)
    assert artifact_retention(0.0, 10.0, 15.0) == pytest.approx(1.5)
    assert artifact_retention(1.0, 1.0, 5.0) is None
    assert artifact_retention(1.0, 1.0 + 5e-7, 2.0) is None


def test_propagation_quality_pairs_the_two_psnr_references() -> None:
    clean = np.full((2, 16, 16), 128, dtype=np.uint8)
    degraded = clean.copy()
    degraded[:, :, 8:] = 0
    recon_clean = clean.copy()
    kept = propagation_quality(clean, degraded, degraded)
    assert kept.psnr_recon_degraded == float("inf")
    assert kept.psnr_recon_clean == pytest.approx(psnr(clean, degraded))
    assert kept.blocking_retention == pytest.approx(1.0)
    removed = propagation_quality(clean, degraded, recon_clean)
    assert removed.psnr_recon_clean == float("inf")
    assert removed.blocking_retention == pytest.approx(0.0)


def test_gpu_seconds_counts_one_clean_encode_and_nine_pairs() -> None:
    assert propagation_gpu_seconds(1.0, 1.0) == pytest.approx(34 * (1.0 + 9 * 2.0))
    assert propagation_gpu_seconds(4.0, 6.0, clips=2, variants=1) == pytest.approx(2 * (4.0 + 10.0))
    assert format_duration(57.14 * 60) == "57 min"
    assert format_duration(4.72 * 60) == "5 min"
    assert format_duration(2.168 * 3600) == "2.2 h"


def test_l4_estimate_matches_the_reconstruction_table() -> None:
    root = Path(__file__).resolve().parents[1]
    table = l4_runtime_markdown(root / "docs" / "results" / "vae-recon.csv")
    note = (root / "docs" / "learn" / "10-artifact-propagation.md").read_text(encoding="utf-8")
    payload = json.loads(
        (root / "notebooks" / "artifact_propagation.ipynb").read_text(encoding="utf-8")
    )
    notebook = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    assert table in note
    assert table in notebook
    assert "| wan2_1 | 4.02 s | 6.73 s | 57 min |" in table
    assert "| hunyuan | 8.70 s | 15.83 s | 2.2 h |" in table
    assert "| ltx | 0.42 s | 0.46 s | 5 min |" in table
    assert "| wan2_2 | 2.02 s | 7.47 s | 50 min |" in table


def test_pull_and_push_round_trip_without_creating_a_missing_root(tmp_path: Path) -> None:
    local = tmp_path / "local.csv"
    local.write_text("keep", encoding="utf-8")
    missing = tmp_path / "no-drive" / "results" / "out.csv"
    assert pull_resume(local, missing, timeout=5) == "absent"
    assert local.read_text(encoding="utf-8") == "keep"
    assert push_copy(local, missing, timeout=5, require_dir=tmp_path / "no-drive") == "skipped"
    assert not missing.exists()
    assert not (tmp_path / "no-drive").exists()

    root = tmp_path / "drive"
    root.mkdir()
    remote = root / "results" / "out.csv"
    remote.parent.mkdir()
    remote.write_text("from-drive", encoding="utf-8")
    assert pull_resume(local, remote, timeout=5) == "copied"
    assert local.read_text(encoding="utf-8") == "from-drive"
    local.write_text("from-local", encoding="utf-8")
    assert push_copy(local, remote, timeout=5, require_dir=root) == "copied"
    assert remote.read_text(encoding="utf-8") == "from-local"
    assert directory_state(root, timeout=5) == "present"
    assert directory_state(tmp_path / "absent", timeout=5) == "absent"


def test_stage_files_skips_files_already_local(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    remote.mkdir()
    local.mkdir()
    (remote / "a.y4m").write_text("aaa", encoding="utf-8")
    (remote / "b.y4m").write_text("bbb", encoding="utf-8")
    (remote / "note.txt").write_text("no", encoding="utf-8")
    (local / "a.y4m").write_text("already", encoding="utf-8")
    first = stage_files(remote, local, suffix=".y4m", timeout=5)
    assert first.status == "ok"
    assert first.copied == 1
    assert first.present == 1
    assert (local / "a.y4m").read_text(encoding="utf-8") == "already"
    assert (local / "b.y4m").read_text(encoding="utf-8") == "bbb"
    assert not (local / "note.txt").exists()
    absent = stage_files(tmp_path / "missing", local, suffix=".y4m", timeout=5)
    assert absent.status == "absent"
    assert absent.copied == 0


def test_child_timeout_kills_the_process() -> None:
    code, text = handoff_child("import time; time.sleep(30)", [], 0.2)
    assert code is None
    assert text == ""


def _write_clip(path: Path, *, seed: int, frames: int = 4, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    luma = rng.integers(0, 256, size=(frames, size, size), dtype=np.uint8)
    chroma = rng.integers(0, 256, size=(frames, size // 2, size // 2), dtype=np.uint8)
    write_y4m(path, luma, chroma, chroma, fps_num=25, fps_den=1)


class _FakeCodec:
    def __init__(self) -> None:
        self.encodes = 0
        self.decodes = 0

    def encode(self, y4m: Path) -> tuple[np.ndarray, float]:
        self.encodes += 1
        _info, luma, _u, _v = read_y4m(y4m)
        return luma.astype(np.float64), 0.01

    def decode_luma(self, latent: np.ndarray) -> tuple[np.ndarray, float]:
        self.decodes += 1
        return np.clip(np.rint(latent), 0, 255).astype(np.uint8), 0.02


def test_run_resumes_and_pushes_after_each_new_row(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips"
    degraded = tmp_path / "degraded"
    clip_dir.mkdir()
    _write_clip(clip_dir / "alpha.y4m", seed=1)
    _write_clip(degraded / "alpha__libx264_crf18.y4m", seed=2)
    _write_clip(degraded / "alpha__libx264_crf30.y4m", seed=3)
    out = tmp_path / "results" / "artifact-propagation.csv"
    pushed: list[Path] = []

    def _push(path: Path) -> str:
        pushed.append(path)
        return "copied"

    codec = _FakeCodec()
    points = (("libx264", 18), ("libx264", 30))
    rows = run_propagation(
        clip_dir=clip_dir,
        degraded_dir=degraded,
        out=out,
        opener=lambda _name: codec,
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        push=_push,
        vaes=("wan2_1",),
        points=points,
    )
    assert codec.encodes == 3
    assert codec.decodes == 2
    assert len(rows) == 2
    assert len(pushed) == 2
    assert rows[0].preset == BENCHMARK_PRESET["libx264"]
    again = _FakeCodec()
    second = run_propagation(
        clip_dir=clip_dir,
        degraded_dir=degraded,
        out=out,
        opener=lambda _name: again,
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        push=_push,
        vaes=("wan2_1",),
        points=points,
    )
    assert again.encodes == 0
    assert len(second) == 2
    assert len(pushed) == 2
    loaded = read_propagation(out)
    assert len(loaded) == 2
    assert loaded[0].latent_mse == pytest.approx(rows[0].latent_mse)
    text = out.read_text(encoding="utf-8")
    changed = tmp_path / "other.csv"
    changed.write_text(text.replace(loaded[0].config_sha256, "0" * 64), encoding="utf-8")
    with pytest.raises(Exception, match="different ladder"):
        run_propagation(
            clip_dir=clip_dir,
            degraded_dir=degraded,
            out=changed,
            opener=lambda _name: _FakeCodec(),
            runtime_dtype="bfloat16",
            gpu_name="NVIDIA L4",
            vaes=("wan2_1",),
            points=points,
        )


def test_run_refuses_a_missing_degraded_source(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    _write_clip(clip_dir / "alpha.y4m", seed=1)
    codec = _FakeCodec()
    with pytest.raises(Exception, match="missing 1 degraded"):
        run_propagation(
            clip_dir=clip_dir,
            degraded_dir=tmp_path / "degraded",
            out=tmp_path / "out.csv",
            opener=lambda _name: codec,
            runtime_dtype="bfloat16",
            gpu_name="cpu",
            vaes=("wan2_1",),
            points=(("libx264", 18),),
        )
    assert codec.encodes == 0


def _has_encoder(name: str) -> bool:
    try:
        binary = ffmpeg_binary()
    except RuntimeError:
        return False
    completed = subprocess.run(
        [binary, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    return name in completed.stdout


@pytest.mark.skipif(not _has_encoder("libx264"), reason="ffmpeg has no libx264")
def test_degrade_writes_pixels_and_skips_a_finished_file(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    source = clip_dir / "tiny.y4m"
    rng = np.random.default_rng(0)
    luma = rng.integers(0, 256, size=(6, 64, 64), dtype=np.uint8)
    chroma = rng.integers(0, 256, size=(6, 32, 32), dtype=np.uint8)
    write_y4m(source, luma, chroma, chroma, fps_num=25, fps_den=1)
    out_dir = tmp_path / "degraded"
    points = (("libx264", 18), ("libx264", 38))
    presets = {"libx264": "ultrafast"}
    first = degrade_sources(clip_dir, out_dir, presets=presets, points=points)
    assert [row.skipped for row in first] == [False, False]
    assert first[0].frames == 6
    fine, _u, _v = read_y4m(first[0].y4m)[1:]
    coarse = read_y4m(first[1].y4m)[1]
    assert psnr(luma, fine) > psnr(luma, coarse)
    stale = first[0].y4m
    stale.write_bytes(b"not a y4m")
    redone = degrade_one(source, stale, codec="libx264", crf=18, preset="ultrafast")
    assert redone.skipped is False
    assert read_y4m(stale)[1].shape == luma.shape
    second = degrade_sources(clip_dir, out_dir, presets=presets, points=points)
    assert all(row.skipped for row in second)


def test_degrade_script_forwards_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "degrade-sources.py"
    spec = importlib.util.spec_from_file_location("degrade_sources_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen: dict[str, object] = {}

    def fake(clip_dir: Path, out_dir: Path, *, only: set[str] | None = None) -> list[object]:
        seen["clip_dir"] = clip_dir
        seen["out_dir"] = out_dir
        seen["only"] = only
        return []

    monkeypatch.setattr(module, "degrade_sources", fake)
    assert (
        module.main(
            ["--clip-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--only", "aspen"]
        )
        == 0
    )
    assert seen["only"] == {"aspen"}


def _require_vae() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        if os.environ.get("LATENTCODECBENCH_REQUIRE_VAE") == "1":
            pytest.fail("CI requires the vae extra; torch is not installed")
        pytest.importorskip("torch")
    raise AssertionError("unreachable")


def test_tiny_session_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _require_vae()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from latentcodecbench.vaes.propagate import VideoVaeSession, open_propagation_vae
    from latentcodecbench.vaes.tiny import build_tiny

    source = tmp_path / "clip.y4m"
    _write_clip(source, seed=4, frames=9, size=32)
    session = VideoVaeSession(build_tiny("wan2_1"))
    latent, encode_seconds = session.encode(source)
    assert encode_seconds > 0
    assert latent.shape[0] == 1
    assert latent.shape[1] == 16
    luma, decode_seconds = session.decode_luma(latent)
    assert decode_seconds > 0
    assert luma.shape == (9, 32, 32)
    assert luma.dtype == np.uint8

    hunyuan = build_tiny("hunyuan")
    assert hunyuan.module.use_tiling is False

    def fake_load(name: str, *, dtype: object, device: str) -> object:
        assert name == "hunyuan"
        assert dtype == "bf16"
        assert device == "cpu"
        return hunyuan

    monkeypatch.setattr("latentcodecbench.vaes.propagate.load_vae", fake_load)
    opened = open_propagation_vae("hunyuan", dtype="bf16", device="cpu")
    assert opened.vae is hunyuan
    assert hunyuan.module.use_tiling is True


def _notebook_cells() -> list[dict[str, Any]]:
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "artifact_propagation.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    return list(payload["cells"])


def _code(cells: list[dict[str, Any]], needle: str) -> str:
    for cell in cells:
        source = "".join(cell["source"])
        if needle in source and cell["cell_type"] == "code":
            return source
    raise AssertionError(f"notebook has no code cell containing {needle}")


def test_notebook_keeps_results_local_and_copies_drive_separately() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "notebooks" / "artifact_propagation.ipynb").read_text(encoding="utf-8")
    cells = _notebook_cells()
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "drive.mount" not in code
    assert "pull_resume" not in code
    assert "push_copy" not in code
    assert "stage_files" not in code
    assert "/content/drive" not in text
    assert "/content/results/artifact-propagation.csv" in text
    assert "/content/latentcodecbench/drive-credentials.json" in text
    assert "artifact-propagation.csv" in text
    assert "canonical-clips.tar" in text
    assert "LatentCodecBench" in text
    install = next(
        "".join(cell["source"])
        for cell in code_cells
        if "diffusers==0.41.0" in "".join(cell["source"])
    )
    assert "google-api-python-client" not in install
    assert "google-auth" not in install
    assert "resume_local" in code
    assert "push_local_csv" in code
    assert "refresh_colab_credentials" in code
    assert "shuvam1024/phase-3-artifact-propagation" in text
    assert "files.download" in code
    needles = [
        "diffusers==0.41.0",
        "FFMPEG_SHA256",
        "authenticate_user",
        "degrade_sources(",
        "resume_local",
        "status_lines",
        "run_propagation(",
        "files.download",
    ]
    indexes = [
        next(index for index, cell in enumerate(code_cells) if needle in "".join(cell["source"]))
        for needle in needles
    ]
    assert indexes == sorted(indexes)
    assert len(set(indexes)) == len(indexes)
    auth = next(
        "".join(cell["source"])
        for cell in code_cells
        if "authenticate_user" in "".join(cell["source"])
    )
    tree = ast.parse(auth)
    top_level = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "google.colab.auth"
    ]
    assert top_level == []
    nested = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom) and child.module == "google.colab.auth":
                nested = True
    assert nested
    drive_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    default_call = next(
        node
        for node in drive_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "default"
    )
    refresh_call = next(
        node
        for node in drive_calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "refresh"
    )
    scope_kw = next(keyword for keyword in default_call.keywords if keyword.arg == "scopes")
    assert isinstance(scope_kw.value, ast.List)
    assert len(scope_kw.value.elts) == 1
    scope_name = scope_kw.value.elts[0]
    assert isinstance(scope_name, ast.Name) and scope_name.id == "DRIVE_SCOPE"
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_authorized_user"
    ]
    assert default_call.lineno < refresh_call.lineno < writes[0].lineno
    sweep = next(
        "".join(cell["source"]) for cell in code_cells if "def loader" in "".join(cell["source"])
    )
    assert "push_local_csv" in sweep
    assert "refresh=refresh_colab_credentials" in sweep
    assert sweep.index("push(LOCAL_RESULTS)") < sweep.index("run_propagation(")
    script = (root / "scripts" / "install-ffmpeg.sh").read_text(encoding="utf-8")
    sha = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in script.splitlines()
        if line.startswith("sha256=")
    )
    assert sha in text
    assert "LATENTCODECBENCH_FFMPEG" in text


def test_sweep_cell_calls_match_the_package() -> None:
    from latentcodecbench.vaes.propagate import open_propagation_vae
    from latentcodecbench.vaes.smoke import runtime_dtype, runtime_dtype_name

    source = _code(_notebook_cells(), "def loader")
    tree = ast.parse(source)
    signatures = {
        "runtime_dtype": inspect.signature(runtime_dtype),
        "runtime_dtype_name": inspect.signature(runtime_dtype_name),
        "open_propagation_vae": inspect.signature(open_propagation_vae),
    }
    found: dict[str, list[ast.Call]] = {name: [] for name in signatures}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in signatures:
            continue
        found[node.func.id].append(node)
        keywords: dict[str, None] = {}
        for keyword in node.keywords:
            assert keyword.arg is not None
            keywords[keyword.arg] = None
        signatures[node.func.id].bind(*([None] * len(node.args)), **keywords)
    assert found["runtime_dtype"] and found["runtime_dtype_name"] and found["open_propagation_vae"]
    for call in found["runtime_dtype"] + found["runtime_dtype_name"]:
        assert len(call.args) == 1 and not call.keywords
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "torch"
    for call in found["open_propagation_vae"]:
        call_keywords: dict[str, ast.expr] = {}
        for keyword in call.keywords:
            assert keyword.arg is not None
            call_keywords[keyword.arg] = keyword.value
        dtype_node = call_keywords["dtype"]
        assert isinstance(dtype_node, ast.Name) and dtype_node.id == "dtype"
        device = call_keywords["device"]
        assert isinstance(device, ast.Constant) and device.value == "cuda"


def test_sweep_loader_opens_with_the_runtime_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_vae()
    from latentcodecbench.vaes.propagate import open_propagation_vae
    from latentcodecbench.vaes.smoke import runtime_dtype, runtime_dtype_name
    from latentcodecbench.vaes.spec import GEOMETRIES
    from latentcodecbench.vaes.wrapper import VideoVae

    source = _code(_notebook_cells(), "def loader")
    tree = ast.parse(source)

    class _Cuda:
        def is_available(self) -> bool:
            return True

        def get_device_capability(self) -> tuple[int, int]:
            return (8, 9)

    class _Torch:
        cuda = _Cuda()
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

    torch = _Torch()
    dtype: Any = None
    dtype_name: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        segment = ast.get_source_segment(source, node)
        assert segment is not None
        if node.func.id == "runtime_dtype":
            dtype = eval(segment, {"runtime_dtype": runtime_dtype, "torch": torch})
        if node.func.id == "runtime_dtype_name":
            dtype_name = eval(segment, {"runtime_dtype_name": runtime_dtype_name, "torch": torch})
    assert dtype == "bf16"
    assert dtype_name == "bfloat16"

    loader_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "loader"
    )
    loader_source = ast.get_source_segment(source, loader_node)
    assert loader_source is not None
    namespace: dict[str, Any] = {"open_propagation_vae": open_propagation_vae, "dtype": dtype}
    exec(loader_source, namespace)

    seen: dict[str, Any] = {}

    class _Parameter:
        dtype = "bf16"
        device = "cpu"

    class _Module:
        use_tiling = False

        def eval(self) -> _Module:
            return self

        def parameters(self) -> Any:
            return iter([_Parameter()])

        def to(self, device: str) -> _Module:
            seen["device"] = device
            return self

    def fake_load(name: str, *, dtype: Any, device: str) -> VideoVae:
        seen["name"] = name
        seen["dtype"] = dtype
        seen["device"] = device
        return VideoVae(GEOMETRIES[name], _Module())

    monkeypatch.setattr("latentcodecbench.vaes.propagate.load_vae", fake_load)
    namespace["loader"]("wan2_1")
    assert seen["dtype"] == "bf16"
    assert seen["device"] == "cuda"
    assert seen["name"] == "wan2_1"
