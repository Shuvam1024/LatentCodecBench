"""The source-CRF probe: features, the encode-only sweep, and the fit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from latentcodecbench.analysis.probe import (
    CLEAN_CODEC,
    ProbeError,
    ProbeRow,
    file_sha256,
    read_probe,
    reconcile_with_drive,
    run_probe,
)
from latentcodecbench.analysis.probe_features import (
    FEATURE_COLUMNS,
    probe_feature_vector,
)
from latentcodecbench.analysis.probe_fit import (
    NOTE_END,
    NOTE_START,
    ProbeEstimate,
    bootstrap_mean,
    fit_probe,
    leave_clip_folds,
    probe_report,
)
from latentcodecbench.cli import main
from latentcodecbench.data.y4m import read_y4m, write_y4m

ROOT = Path(__file__).resolve().parents[1]


def test_channel_stats_pad_and_a_clean_delta_is_zero() -> None:
    volume = np.zeros((2, 1, 1, 2), dtype=np.float64)
    volume[0, 0, 0, :] = (1.0, 3.0)
    volume[1, 0, 0, :] = (5.0, 5.0)
    luma = np.zeros((2, 16, 16), dtype=np.uint8)
    vector = probe_feature_vector(volume, volume, luma)
    batched = probe_feature_vector(volume[None, ...], volume, luma)
    assert vector.shape == (len(FEATURE_COLUMNS),)
    assert batched == pytest.approx(vector)
    assert vector[FEATURE_COLUMNS.index("latent_mean_c000")] == pytest.approx(2.0)
    assert vector[FEATURE_COLUMNS.index("latent_std_c000")] == pytest.approx(1.0)
    assert vector[FEATURE_COLUMNS.index("latent_abs_mean_c000")] == pytest.approx(2.0)
    assert vector[FEATURE_COLUMNS.index("latent_mean_c001")] == pytest.approx(5.0)
    assert vector[FEATURE_COLUMNS.index("latent_std_c001")] == pytest.approx(0.0)
    assert vector[FEATURE_COLUMNS.index("latent_mean_c002")] == pytest.approx(0.0)
    assert vector[FEATURE_COLUMNS.index("latent_hf_energy")] == pytest.approx(2.0)
    assert vector[FEATURE_COLUMNS.index("delta_mse")] == pytest.approx(0.0)
    assert vector[FEATURE_COLUMNS.index("delta_relative_l2")] == pytest.approx(0.0)
    assert vector[FEATURE_COLUMNS.index("delta_cosine")] == pytest.approx(1.0)
    assert vector[FEATURE_COLUMNS.index("pixel_luma_mean")] == pytest.approx(0.0)
    wide = np.zeros((129, 1, 1, 2))
    with pytest.raises(ValueError, match="128"):
        probe_feature_vector(wide, wide, luma)


def test_leave_clip_out_does_not_train_on_the_held_out_clip() -> None:
    clips = ["b", "a", "b", "a", "c", "c"]
    folds = leave_clip_folds(clips)
    assert [clips[int(test[0])] for _train, test in folds] == ["a", "b", "c"]
    for train, test in folds:
        assert set(np.array(clips)[train]).isdisjoint(set(np.array(clips)[test]))
        assert len(train) + len(test) == len(clips)


def test_bootstrap_of_identical_clips_has_no_width() -> None:
    estimate, low, high = bootstrap_mean(np.array([2.0, 2.0, 2.0]), draws=50, seed=0)
    assert estimate == pytest.approx(2.0)
    assert low == pytest.approx(estimate)
    assert high == pytest.approx(estimate)
    spread, low_spread, high_spread = bootstrap_mean(np.array([1.0, 2.0, 3.0]), draws=200, seed=0)
    assert low_spread < spread < high_spread
    assert low_spread >= 1.0
    assert high_spread <= 3.0


def test_latent_feature_beats_chance_and_the_pixel_baseline(tmp_path: Path) -> None:
    rows = [_row(clip, crf) for clip in ("a", "b", "c", "d") for crf in (18, 30, 38)]
    estimates = fit_probe(rows, draws=40, seed=0)
    latent = _one(estimates, "regression", "latent")
    pixel = _one(estimates, "regression", "pixel")
    chance = _one(estimates, "regression", "chance")
    assert latent.estimate < 1.0
    assert chance.estimate == pytest.approx(7.111111, abs=1e-3)
    assert pixel.estimate == pytest.approx(chance.estimate)
    assert latent.ci_low == pytest.approx(latent.estimate)
    assert latent.ci_high == pytest.approx(latent.estimate)
    classified = _one(estimates, "classification", "latent")
    chance_class = _one(estimates, "classification", "chance")
    assert classified.estimate > chance_class.estimate
    assert chance_class.estimate == pytest.approx(1.0 / 3.0)
    with pytest.raises(ProbeError, match="two clips"):
        fit_probe(rows[:3])

    note = tmp_path / "note.md"
    note.write_text(f"before\n{NOTE_START}\n\nnone yet\n\n{NOTE_END}\nafter\n", encoding="utf-8")
    figure = tmp_path / "probe.png"
    table = tmp_path / "probe.csv"
    code = main(
        [
            "probe",
            "fit",
            "--features",
            str(_write_features(tmp_path / "features.csv", rows)),
            "--out",
            str(table),
            "--figure",
            str(figure),
            "--note",
            str(note),
        ]
    )
    assert code == 0
    text = note.read_text(encoding="utf-8")
    assert text.startswith("before\n")
    assert text.endswith("after\n")
    assert probe_report(fit_probe(rows)) in text
    assert figure.stat().st_size > 1000
    assert "latent" in table.read_text(encoding="utf-8")


def test_run_encodes_without_decoding_and_resumes(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips"
    degraded = tmp_path / "degraded"
    clip_dir.mkdir()
    _write_clip(clip_dir / "alpha.y4m", seed=1)
    _write_clip(degraded / "alpha__libx264_crf18.y4m", seed=2)
    _write_clip(degraded / "alpha__libx264_crf30.y4m", seed=3)
    out = tmp_path / "results" / "latent-probe-features.csv"
    pushed: list[Path] = []

    def _push(path: Path) -> str:
        pushed.append(path)
        return "ok"

    codec = _FakeEncoder()
    points = (("libx264", 18), ("libx264", 30))
    rows = run_probe(
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
    assert len(rows) == 3
    assert len(pushed) == 3
    clean = next(row for row in rows if row.source_codec == CLEAN_CODEC)
    assert clean.source_crf is None
    assert clean.features[FEATURE_COLUMNS.index("delta_mse")] == pytest.approx(0.0)
    assert clean.source_sha256 == file_sha256(clip_dir / "alpha.y4m")
    degraded_row = next(row for row in rows if row.source_crf == 18)
    assert (
        degraded_row.source_sha256
        == hashlib.sha256((degraded / "alpha__libx264_crf18.y4m").read_bytes()).hexdigest()
    )
    assert degraded_row.preset == "medium"
    again = _FakeEncoder()
    second = run_probe(
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
    assert len(second) == 3
    assert len(pushed) == 3
    loaded = read_probe(out)
    assert len(loaded) == 3
    short = tmp_path / "short.csv"
    short.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    assert reconcile_with_drive(short, out) == "ok"
    changed = tmp_path / "other.csv"
    text = out.read_text(encoding="utf-8").replace(loaded[0].config_sha256, "0" * 64)
    changed.write_text(text, encoding="utf-8")
    assert reconcile_with_drive(out, changed) == "refused-config"
    assert len(read_probe(out)) == 3
    with pytest.raises(ProbeError, match="different feature schema"):
        run_probe(
            clip_dir=clip_dir,
            degraded_dir=degraded,
            out=changed,
            opener=lambda _name: _FakeEncoder(),
            runtime_dtype="bfloat16",
            gpu_name="NVIDIA L4",
            vaes=("wan2_1",),
            points=points,
        )


def test_learn_note_keeps_the_results_marker() -> None:
    text = (ROOT / "docs" / "learn" / "14-linear-probe.md").read_text(encoding="utf-8")
    assert NOTE_START in text
    assert NOTE_END in text
    assert "has not been measured" in text
    assert "compsr" not in text.lower()


def test_notebook_encodes_only_and_keeps_the_drive_merge() -> None:
    path = ROOT / "notebooks" / "latent_probe_features.ipynb"
    text = path.read_text(encoding="utf-8")
    cells = json.loads(text)["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "drive.mount" not in code
    assert "/content/drive" not in text
    assert "decode_luma" not in code
    assert "canonical-clips.tar" in text
    assert "latent-probe-features.csv" in text
    assert "source_sha256" in text
    assert "shuvam1024/phase-3-3-linear-probe" in text
    assert "reconcile_with_drive" in code
    assert "push_local_csv" in code
    needles = [
        "diffusers==0.41.0",
        "FFMPEG_SHA256",
        "authenticate_user",
        "degrade_sources(",
        "resume_local",
        "status_lines",
        "run_probe(",
        "files.download",
    ]
    indexes = [
        next(index for index, cell in enumerate(code_cells) if needle in "".join(cell["source"]))
        for needle in needles
    ]
    assert indexes == sorted(indexes)
    assert len(set(indexes)) == len(indexes)
    sweep = next(
        "".join(cell["source"]) for cell in code_cells if "def loader" in "".join(cell["source"])
    )
    assert "decode" not in sweep
    tree = ast.parse(sweep)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "run_probe" in calls
    assert "run_propagation" not in calls


class _FakeEncoder:
    def __init__(self) -> None:
        self.encodes = 0

    def encode(self, y4m: Path) -> tuple[np.ndarray, float]:
        self.encodes += 1
        _info, luma, _u, _v = read_y4m(y4m)
        volume = luma.astype(np.float64)[None, ...]
        return volume, 0.01


def _write_clip(path: Path, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    luma = rng.integers(0, 256, size=(4, 16, 16), dtype=np.uint8)
    chroma = rng.integers(0, 256, size=(4, 8, 8), dtype=np.uint8)
    write_y4m(path, luma, chroma, chroma, fps_num=25, fps_den=1)


def _row(clip: str, crf: int) -> ProbeRow:
    features = [0.0] * len(FEATURE_COLUMNS)
    features[0] = float(crf)
    features[FEATURE_COLUMNS.index("pixel_luma_mean")] = 1.0
    return ProbeRow(
        run_id="run",
        git_sha="a" * 40,
        config_sha256="b" * 64,
        clip_id=clip,
        vae="wan2_1",
        source_codec="libx264",
        source_crf=crf,
        preset="medium",
        source_sha256="c" * 64,
        width=16,
        height=16,
        frames=4,
        latent_shape="1 x 4 x 16 x 16",
        encode_seconds=0.01,
        runtime_dtype="bfloat16",
        gpu_name="NVIDIA L4",
        features=tuple(features),
    )


def _write_features(path: Path, rows: list[ProbeRow]) -> Path:
    from latentcodecbench.analysis.probe import write_probe

    write_probe(path, rows)
    return path


def _one(estimates: list[ProbeEstimate], task: str, model: str) -> ProbeEstimate:
    found = [row for row in estimates if row.task == task and row.model == model]
    assert len(found) == 1
    return found[0]
