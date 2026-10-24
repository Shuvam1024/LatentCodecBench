"""Leave-clip-out linear probe of source CRF.

The fit is separate for each VAE and each source codec. CRF is not a
shared scale, and a latent channel count is not shared either. Each
fold holds out one clip, so the three rungs of a clip never train the
model that scores them.

Regression is ridge on the CRF. Classification is ridge on the three
rung indicators, and the predicted rung is the largest score. Both use
the same penalty on z-scored columns. Chance uses no features. The
pixel model uses only the luma summary. The interval is a percentile
bootstrap over clips. ADR-0013.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latentcodecbench.analysis.probe import CLEAN_CODEC, ProbeError, ProbeRow
from latentcodecbench.analysis.probe_features import (
    FEATURE_COLUMNS,
    LATENT_FEATURE_COLUMNS,
    PIXEL_FEATURE_COLUMNS,
)
from latentcodecbench.codecs.degrade import STUDY_LADDER
from latentcodecbench.vaes.checkpoints import SMOKE_ORDER

RIDGE_LAMBDA = 1.0
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 0
NOTE_START = "<!-- probe-results:start -->"
NOTE_END = "<!-- probe-results:end -->"

RESULT_COLUMNS: tuple[str, ...] = (
    "vae",
    "source_codec",
    "task",
    "model",
    "metric",
    "clips",
    "estimate",
    "ci_low",
    "ci_high",
    "ridge_lambda",
    "bootstrap_draws",
)

_CODEC_LABEL = {"libx264": "x264", "libx265": "x265", "libsvtav1": "SVT-AV1"}
_MODEL_ORDER = ("latent", "pixel", "chance")
_TASK_ORDER = ("regression", "classification")
_MIN_STD = 1e-8


@dataclass(frozen=True)
class ProbeEstimate:
    """One model on one VAE and codec, with a clip-bootstrap interval."""

    vae: str
    source_codec: str
    task: str
    model: str
    metric: str
    clips: int
    estimate: float
    ci_low: float
    ci_high: float
    ridge_lambda: float
    bootstrap_draws: int


def leave_clip_folds(clip_ids: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train and test indices, one held-out clip at a time.

    The test indices are exactly the rows of that clip. The train indices
    are every other row.
    """
    ids = np.asarray(clip_ids)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for name in sorted(set(clip_ids)):
        test = np.flatnonzero(ids == name)
        train = np.flatnonzero(ids != name)
        folds.append((train, test))
    return folds


def bootstrap_mean(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean, and the 2.5 and 97.5 percentiles of bootstrap means.

    One clip has no interval, so the bounds equal the estimate. ``draws``
    resamples clips with replacement.
    """
    if values.size == 0:
        raise ProbeError("bootstrap needs a clip")
    estimate = float(values.mean())
    if values.size == 1 or draws < 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(draws, values.size))
    means = values[picks].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return estimate, float(low), float(high)


def fit_probe(
    rows: list[ProbeRow],
    *,
    lam: float = RIDGE_LAMBDA,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> list[ProbeEstimate]:
    """Fit every VAE and codec that has at least two clips.

    Clean rows are the stored reference features. They are not targets.
    A group with one clip is skipped. No group at all is an error.
    """
    degraded = [row for row in rows if row.source_codec != CLEAN_CODEC]
    if not degraded:
        raise ProbeError("no degraded probe rows")
    grouped: dict[tuple[str, str], list[ProbeRow]] = defaultdict(list)
    for row in degraded:
        if row.source_crf is None:
            raise ProbeError(f"degraded row {row.clip_id} {row.vae} has no CRF")
        grouped[(row.vae, row.source_codec)].append(row)

    ordered: list[tuple[str, str]] = []
    for vae in SMOKE_ORDER:
        for codec in STUDY_LADDER:
            if (vae, codec) in grouped:
                ordered.append((vae, codec))
    for key in sorted(grouped):
        if key not in ordered:
            ordered.append(key)

    estimates: list[ProbeEstimate] = []
    for key in ordered:
        block = grouped[key]
        if len({row.clip_id for row in block}) < 2:
            continue
        estimates.extend(_fit_group(block, lam=lam, draws=draws, seed=seed))
    if not estimates:
        raise ProbeError("need at least two clips in one VAE and codec")
    return estimates


def probe_report(estimates: list[ProbeEstimate]) -> str:
    """Markdown table. Regression is MAE in CRF points. Classification is accuracy."""
    if not estimates:
        raise ProbeError("no probe estimates")
    lines = [
        "Leave-clip-out ridge, one VAE and one source codec at a time. "
        "Regression is mean absolute error in CRF points of that codec. "
        "Classification is accuracy on the three rungs of the study ladder. "
        "The interval is a 95% bootstrap over clips.",
        "",
        "| VAE | Source | Task | Model | Clips | Estimate | 95% CI |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in estimates:
        label = _CODEC_LABEL.get(row.source_codec, row.source_codec)
        span = f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        lines.append(
            f"| {row.vae} | {label} | {row.task} | {row.model} | {row.clips} | "
            f"{row.estimate:.3f} | {span} |"
        )
    return "\n".join(lines) + "\n"


def write_probe_table(path: Path, estimates: list[ProbeEstimate]) -> None:
    """Write the estimate table. The figure and the note are separate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in estimates:
            writer.writerow(
                {
                    "vae": row.vae,
                    "source_codec": row.source_codec,
                    "task": row.task,
                    "model": row.model,
                    "metric": row.metric,
                    "clips": str(row.clips),
                    "estimate": repr(row.estimate),
                    "ci_low": repr(row.ci_low),
                    "ci_high": repr(row.ci_high),
                    "ridge_lambda": repr(row.ridge_lambda),
                    "bootstrap_draws": str(row.bootstrap_draws),
                }
            )


def plot_probe(estimates: list[ProbeEstimate], path: Path) -> None:
    """Grouped bars of MAE and accuracy, with the bootstrap interval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: list[tuple[str, str]] = []
    for row in estimates:
        key = (row.vae, row.source_codec)
        if key not in groups:
            groups.append(key)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(6.0, len(groups) * 0.85), 4.0),
        constrained_layout=True,
    )
    panels = (
        (axes[0], "regression", "MAE (CRF points)"),
        (axes[1], "classification", "Accuracy"),
    )
    offsets = {"latent": -0.25, "pixel": 0.0, "chance": 0.25}
    for axis, task, ylabel in panels:
        for model in _MODEL_ORDER:
            heights: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for key in groups:
                match = next(
                    row
                    for row in estimates
                    if row.task == task
                    and row.model == model
                    and (row.vae, row.source_codec) == key
                )
                heights.append(match.estimate)
                lows.append(max(0.0, match.estimate - match.ci_low))
                highs.append(max(0.0, match.ci_high - match.estimate))
            positions = np.arange(len(groups), dtype=np.float64) + offsets[model]
            axis.bar(
                positions,
                heights,
                width=0.25,
                label=model,
                yerr=np.vstack([lows, highs]),
                capsize=2,
            )
        axis.set_xticks(np.arange(len(groups)))
        axis.set_xticklabels(
            [f"{vae}\n{_CODEC_LABEL.get(codec, codec)}" for vae, codec in groups],
            fontsize=8,
        )
        axis.set_ylabel(ylabel)
        axis.set_title(task)
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def replace_note_section(path: Path, report: str) -> None:
    """Replace the marked results section of a learn note."""
    text = path.read_text(encoding="utf-8")
    if NOTE_START not in text or NOTE_END not in text:
        raise ProbeError(f"{path} has no probe-results markers")
    start = text.index(NOTE_START) + len(NOTE_START)
    end = text.index(NOTE_END)
    if end < start:
        raise ProbeError("probe-results markers are out of order")
    updated = text[:start] + "\n\n" + report.rstrip() + "\n\n" + text[end:]
    path.write_text(updated, encoding="utf-8")


def write_probe_outputs(
    rows: list[ProbeRow],
    *,
    table: Path,
    figure: Path,
    note: Path | None = None,
    lam: float = RIDGE_LAMBDA,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> str:
    """Fit ``rows`` and write the table, the figure, and an optional note.

    Returns the markdown table.
    """
    estimates = fit_probe(rows, lam=lam, draws=draws, seed=seed)
    report = probe_report(estimates)
    write_probe_table(table, estimates)
    plot_probe(estimates, figure)
    if note is not None:
        replace_note_section(note, report)
    return report


def _fit_group(
    rows: list[ProbeRow],
    *,
    lam: float,
    draws: int,
    seed: int,
) -> list[ProbeEstimate]:
    codec = rows[0].source_codec
    vae = rows[0].vae
    clip_ids = [row.clip_id for row in rows]
    crf_values: list[float] = []
    rung_values: list[int] = []
    for row in rows:
        if row.source_crf is None:
            raise ProbeError(f"degraded row {row.clip_id} {row.vae} has no CRF")
        crf_values.append(float(row.source_crf))
        rung_values.append(_rung(codec, row.source_crf))
    crf = np.array(crf_values, dtype=np.float64)
    rung = np.array(rung_values, dtype=np.int64)
    matrix = np.array([row.features for row in rows], dtype=np.float64)
    if matrix.shape[1] != len(FEATURE_COLUMNS):
        raise ProbeError(f"feature length {matrix.shape[1]} is not {len(FEATURE_COLUMNS)}")
    latent = matrix[:, : len(LATENT_FEATURE_COLUMNS)]
    pixel = matrix[:, len(LATENT_FEATURE_COLUMNS) :]
    if pixel.shape[1] != len(PIXEL_FEATURE_COLUMNS):
        raise ProbeError("pixel feature length does not match the schema")

    pred_crf = {model: np.zeros(len(rows), dtype=np.float64) for model in _MODEL_ORDER}
    pred_rung = {model: np.zeros(len(rows), dtype=np.int64) for model in _MODEL_ORDER}
    for train, test in leave_clip_folds(clip_ids):
        pred_crf["chance"][test] = float(crf[train].mean())
        pred_rung["chance"][test] = _chance_class(rung[train])
        for model, block in (("latent", latent), ("pixel", pixel)):
            z_train, z_test = _standardize(block[train], block[test])
            pred_crf[model][test] = _ridge_predict(z_train, crf[train], z_test, lam)
            class_scores = [
                _ridge_predict(z_train, (rung[train] == label).astype(np.float64), z_test, lam)
                for label in (0, 1, 2)
            ]
            pred_rung[model][test] = np.argmax(np.column_stack(class_scores), axis=1)

    estimates: list[ProbeEstimate] = []
    for task in _TASK_ORDER:
        metric = "mae" if task == "regression" else "accuracy"
        for model in _MODEL_ORDER:
            scores = _clip_scores(clip_ids, crf, rung, pred_crf[model], pred_rung[model], task)
            estimate, low, high = bootstrap_mean(scores, draws=draws, seed=seed)
            estimates.append(
                ProbeEstimate(
                    vae=vae,
                    source_codec=codec,
                    task=task,
                    model=model,
                    metric=metric,
                    clips=int(scores.size),
                    estimate=estimate,
                    ci_low=low,
                    ci_high=high,
                    ridge_lambda=lam,
                    bootstrap_draws=draws,
                )
            )
    return estimates


def _clip_scores(
    clip_ids: list[str],
    crf: np.ndarray,
    rung: np.ndarray,
    pred_crf: np.ndarray,
    pred_rung: np.ndarray,
    task: str,
) -> np.ndarray:
    scores: list[float] = []
    for name in sorted(set(clip_ids)):
        mask = np.array([clip == name for clip in clip_ids])
        if task == "regression":
            scores.append(float(np.mean(np.abs(pred_crf[mask] - crf[mask]))))
        else:
            scores.append(float(np.mean(pred_rung[mask] == rung[mask])))
    return np.array(scores, dtype=np.float64)


def _chance_class(train_rung: np.ndarray) -> int:
    """Most common training rung. A tie goes to the mildest rung."""
    labels = sorted({int(value) for value in train_rung})
    counts = {label: int(np.sum(train_rung == label)) for label in labels}
    return min(labels, key=lambda label: (-counts[label], label))


def _rung(codec: str, crf: int) -> int:
    ladder = STUDY_LADDER.get(codec)
    if ladder is None or crf not in ladder:
        raise ProbeError(f"{codec} CRF {crf} is not a study rung")
    return ladder.index(crf)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[1] == 0:
        return train, test
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    keep = std > _MIN_STD
    if not np.any(keep):
        return np.zeros((train.shape[0], 0)), np.zeros((test.shape[0], 0))
    return (train[:, keep] - mean[keep]) / std[keep], (test[:, keep] - mean[keep]) / std[keep]


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Ridge with an unpenalized intercept. Columns are already z-scored."""
    y_mean = float(train_y.mean())
    if train_x.shape[1] == 0:
        return np.full(test_x.shape[0], y_mean)
    centered = train_y - y_mean
    n_rows, n_cols = train_x.shape
    if n_rows >= n_cols:
        gram = train_x.T @ train_x
        gram.flat[:: n_cols + 1] += lam
        beta = np.linalg.solve(gram, train_x.T @ centered)
    else:
        gram = train_x @ train_x.T
        gram.flat[:: n_rows + 1] += lam
        beta = train_x.T @ np.linalg.solve(gram, centered)
    return test_x @ beta + y_mean
