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
class _ClipScore:
    """One clip's score inside one VAE, codec, task, and model."""

    vae: str
    source_codec: str
    clip_id: str
    model: str
    task: str
    score: float


@dataclass(frozen=True)
class ProbeWriteup:
    """Per-codec fits plus the pooled and caveat sections."""

    per_codec: list[ProbeEstimate]
    per_codec_markdown: str
    pooled_estimates: list[ProbeEstimate]
    clean_estimates: list[ProbeEstimate]
    codec_estimates: list[ProbeEstimate]
    pooled: str
    clean: str
    codec_id: str
    hashes: str

    def note_body(self) -> str:
        """Markdown that replaces the marked section of the learn note."""
        parts = [self.per_codec_markdown.rstrip()]
        for part in (self.pooled, self.clean, self.codec_id, self.hashes):
            if part.strip():
                parts.append(part.rstrip())
        return "\n\n".join(parts) + "\n"


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
    estimates, _scores = _fit_all(rows, lam=lam, draws=draws, seed=seed)
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

    The table and the figure are the per-codec fits. The note also gets
    the pooled rung accuracy and the two extra readouts. Returns that
    markdown.
    """
    writeup = probe_writeup(rows, lam=lam, draws=draws, seed=seed)
    report = writeup.note_body()
    write_probe_table(
        table,
        [
            *writeup.per_codec,
            *writeup.pooled_estimates,
            *writeup.clean_estimates,
            *writeup.codec_estimates,
        ],
    )
    plot_probe(writeup.per_codec, figure)
    if note is not None:
        replace_note_section(note, report)
    return report


def _fit_group(
    rows: list[ProbeRow],
    *,
    lam: float,
    draws: int,
    seed: int,
) -> tuple[list[ProbeEstimate], list[_ClipScore]]:
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
    clip_scores: list[_ClipScore] = []
    for task in _TASK_ORDER:
        metric = "mae" if task == "regression" else "accuracy"
        for model in _MODEL_ORDER:
            named = _named_clip_scores(clip_ids, crf, rung, pred_crf[model], pred_rung[model], task)
            values = np.array([score for _name, score in named], dtype=np.float64)
            estimate, low, high = bootstrap_mean(values, draws=draws, seed=seed)
            estimates.append(
                ProbeEstimate(
                    vae=vae,
                    source_codec=codec,
                    task=task,
                    model=model,
                    metric=metric,
                    clips=int(values.size),
                    estimate=estimate,
                    ci_low=low,
                    ci_high=high,
                    ridge_lambda=lam,
                    bootstrap_draws=draws,
                )
            )
            for clip_id, score in named:
                clip_scores.append(
                    _ClipScore(
                        vae=vae,
                        source_codec=codec,
                        clip_id=clip_id,
                        model=model,
                        task=task,
                        score=score,
                    )
                )
    return estimates, clip_scores


def _named_clip_scores(
    clip_ids: list[str],
    crf: np.ndarray,
    rung: np.ndarray,
    pred_crf: np.ndarray,
    pred_rung: np.ndarray,
    task: str,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for name in sorted(set(clip_ids)):
        mask = np.array([clip == name for clip in clip_ids])
        if task == "regression":
            scores.append((name, float(np.mean(np.abs(pred_crf[mask] - crf[mask])))))
        else:
            scores.append((name, float(np.mean(pred_rung[mask] == rung[mask]))))
    return scores


def probe_writeup(
    rows: list[ProbeRow],
    *,
    lam: float = RIDGE_LAMBDA,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> ProbeWriteup:
    """Per-codec table, pooled rung accuracy, and the two extra readouts.

    Pooling averages clip scores from models that were fit inside one
    codec. It does not fit a CRF across codecs. The clean-versus-degraded
    readout and the codec-identity readout use the same ridge and the
    same leave-clip-out split. A readout that the rows cannot support is
    an empty string.
    """
    estimates, scores = _fit_all(rows, lam=lam, draws=draws, seed=seed)
    pooled_estimates = pool_classification(scores, lam=lam, draws=draws, seed=seed)
    pooled = _accuracy_section(
        "Classification accuracy pooled across codecs. The model was still fit "
        "inside one codec. A clip's score is the mean of its per-codec accuracies, "
        "and the `all` row is that mean taken across VAEs as well. The interval "
        "resamples clips. Chance is the per-codec majority rung, pooled the same way.",
        pooled_estimates,
    )
    clean_rows, _clean_scores = _fit_clean_versus_degraded(rows, lam=lam, draws=draws, seed=seed)
    clean = _accuracy_section(
        "Clean versus degraded, one VAE at a time, on the same ridge and the same "
        "leave-clip-out split. Chance is the majority training class. A tie takes "
        "the clean class. The `all` row averages clip scores across VAEs.",
        clean_rows,
    )
    codec_rows, _codec_scores = _fit_codec_identity(rows, lam=lam, draws=draws, seed=seed)
    codec_id = _accuracy_section(
        "Codec identity on the degraded rows only, three codecs, same ridge and "
        "same leave-clip-out split. The label is the codec, not the CRF. Chance "
        "is the majority training codec, and a tie takes the first codec on the "
        "study ladder. The `all` row averages clip scores across VAEs.",
        codec_rows,
    )
    return ProbeWriteup(
        per_codec=estimates,
        per_codec_markdown=probe_report(estimates),
        pooled_estimates=pooled_estimates,
        clean_estimates=clean_rows,
        codec_estimates=codec_rows,
        pooled=pooled,
        clean=clean,
        codec_id=codec_id,
        hashes=hash_disagreement_sentence(rows),
    )


def compose_probe_report(
    rows: list[ProbeRow],
    *,
    lam: float = RIDGE_LAMBDA,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> str:
    """The marked section of the learn note, including pooled accuracy."""
    return probe_writeup(rows, lam=lam, draws=draws, seed=seed).note_body()


def pool_classification(
    scores: list[_ClipScore],
    *,
    lam: float,
    draws: int,
    seed: int,
) -> list[ProbeEstimate]:
    """Per-VAE and all-VAE means of the per-codec classification scores."""
    classified = [row for row in scores if row.task == "classification"]
    if not classified:
        return []
    seen: list[str] = [name for name in SMOKE_ORDER if any(row.vae == name for row in classified)]
    for name in sorted({row.vae for row in classified}):
        if name not in seen:
            seen.append(name)
    pooled: list[ProbeEstimate] = []
    for name in seen:
        block = [row for row in classified if row.vae == name]
        pooled.extend(
            _pool_models(block, vae=name, source_codec="pooled", lam=lam, draws=draws, seed=seed)
        )
    pooled.extend(
        _pool_models(classified, vae="all", source_codec="pooled", lam=lam, draws=draws, seed=seed)
    )
    return pooled


def hash_disagreement_sentence(rows: list[ProbeRow]) -> str:
    """How often one degraded point has more than one ``source_sha256``.

    The count is over ``(clip, codec, CRF)`` after the clean rows are
    dropped. An empty string means every point agreed, or there was
    nothing to compare.
    """
    grouped: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.source_codec == CLEAN_CODEC or row.source_crf is None:
            continue
        grouped[(row.clip_id, row.source_codec, row.source_crf)].add(row.source_sha256)
    bad = {key: hashes for key, hashes in grouped.items() if len(hashes) > 1}
    if not grouped or not bad:
        return ""
    by_codec: dict[str, int] = defaultdict(int)
    for _clip, codec, _crf in bad:
        by_codec[codec] += 1
    disagreed: list[str] = []
    for codec in STUDY_LADDER:
        count = by_codec.get(codec, 0)
        if count:
            disagreed.append(f"{_CODEC_LABEL.get(codec, codec)} {count}")
    agreed = [
        _CODEC_LABEL.get(codec, codec)
        for codec in STUDY_LADDER
        if by_codec.get(codec, 0) == 0 and any(key[1] == codec for key in grouped)
    ]
    if len(agreed) == 1:
        agreed_text = f" {agreed[0]} agrees."
    elif agreed:
        agreed_text = f" {', '.join(agreed[:-1])} and {agreed[-1]} agree."
    else:
        agreed_text = ""
    return (
        f"`source_sha256` disagrees across VAEs on {len(bad)} of {len(grouped)} "
        f"degraded (clip, codec, CRF) points ({', '.join(disagreed)})."
        f"{agreed_text} The rung model is fit inside one codec, so a within-codec "
        "comparison does not mix those files. A cross-VAE comparison at one of "
        "those points does."
    )


def _fit_all(
    rows: list[ProbeRow],
    *,
    lam: float,
    draws: int,
    seed: int,
) -> tuple[list[ProbeEstimate], list[_ClipScore]]:
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
    scores: list[_ClipScore] = []
    for key in ordered:
        block = grouped[key]
        if len({row.clip_id for row in block}) < 2:
            continue
        group_estimates, group_scores = _fit_group(block, lam=lam, draws=draws, seed=seed)
        estimates.extend(group_estimates)
        scores.extend(group_scores)
    if not estimates:
        raise ProbeError("need at least two clips in one VAE and codec")
    return estimates, scores


def _fit_clean_versus_degraded(
    rows: list[ProbeRow],
    *,
    lam: float,
    draws: int,
    seed: int,
) -> tuple[list[ProbeEstimate], list[_ClipScore]]:
    grouped: dict[str, list[ProbeRow]] = defaultdict(list)
    for row in rows:
        grouped[row.vae].append(row)
    estimates: list[ProbeEstimate] = []
    scores: list[_ClipScore] = []
    names = [name for name in SMOKE_ORDER if name in grouped]
    for name in sorted(grouped):
        if name not in names:
            names.append(name)
    for name in names:
        block = grouped[name]
        clips = {row.clip_id for row in block}
        kinds = {row.source_codec == CLEAN_CODEC for row in block}
        if len(clips) < 2 or kinds != {True, False}:
            continue
        labels = np.array(
            [0 if row.source_codec == CLEAN_CODEC else 1 for row in block],
            dtype=np.int64,
        )
        group_estimates, group_scores = _fit_labels(
            block,
            labels,
            vae=name,
            source_codec="clean-degraded",
            n_classes=2,
            lam=lam,
            draws=draws,
            seed=seed,
        )
        estimates.extend(group_estimates)
        scores.extend(group_scores)
    if scores:
        estimates.extend(
            _pool_models(
                scores,
                vae="all",
                source_codec="clean-degraded",
                lam=lam,
                draws=draws,
                seed=seed,
            )
        )
    return estimates, scores


def _fit_codec_identity(
    rows: list[ProbeRow],
    *,
    lam: float,
    draws: int,
    seed: int,
) -> tuple[list[ProbeEstimate], list[_ClipScore]]:
    order = list(STUDY_LADDER)
    degraded = [
        row for row in rows if row.source_codec in STUDY_LADDER and row.source_codec != CLEAN_CODEC
    ]
    grouped: dict[str, list[ProbeRow]] = defaultdict(list)
    for row in degraded:
        grouped[row.vae].append(row)
    estimates: list[ProbeEstimate] = []
    scores: list[_ClipScore] = []
    names = [name for name in SMOKE_ORDER if name in grouped]
    for name in sorted(grouped):
        if name not in names:
            names.append(name)
    for name in names:
        block = grouped[name]
        codecs = {row.source_codec for row in block}
        if codecs != set(order) or len({row.clip_id for row in block}) < 2:
            continue
        labels = np.array([order.index(row.source_codec) for row in block], dtype=np.int64)
        group_estimates, group_scores = _fit_labels(
            block,
            labels,
            vae=name,
            source_codec="codec-id",
            n_classes=len(order),
            lam=lam,
            draws=draws,
            seed=seed,
        )
        estimates.extend(group_estimates)
        scores.extend(group_scores)
    if scores:
        estimates.extend(
            _pool_models(
                scores,
                vae="all",
                source_codec="codec-id",
                lam=lam,
                draws=draws,
                seed=seed,
            )
        )
    return estimates, scores


def _fit_labels(
    rows: list[ProbeRow],
    labels: np.ndarray,
    *,
    vae: str,
    source_codec: str,
    n_classes: int,
    lam: float,
    draws: int,
    seed: int,
) -> tuple[list[ProbeEstimate], list[_ClipScore]]:
    clip_ids = [row.clip_id for row in rows]
    matrix = np.array([row.features for row in rows], dtype=np.float64)
    if matrix.shape[1] != len(FEATURE_COLUMNS):
        raise ProbeError(f"feature length {matrix.shape[1]} is not {len(FEATURE_COLUMNS)}")
    latent = matrix[:, : len(LATENT_FEATURE_COLUMNS)]
    pixel = matrix[:, len(LATENT_FEATURE_COLUMNS) :]
    pred = {model: np.zeros(len(rows), dtype=np.int64) for model in _MODEL_ORDER}
    for train, test in leave_clip_folds(clip_ids):
        pred["chance"][test] = _chance_class(labels[train])
        for model, block in (("latent", latent), ("pixel", pixel)):
            z_train, z_test = _standardize(block[train], block[test])
            class_scores = [
                _ridge_predict(
                    z_train,
                    (labels[train] == label).astype(np.float64),
                    z_test,
                    lam,
                )
                for label in range(n_classes)
            ]
            pred[model][test] = np.argmax(np.column_stack(class_scores), axis=1)
    estimates: list[ProbeEstimate] = []
    scores: list[_ClipScore] = []
    for model in _MODEL_ORDER:
        named: list[tuple[str, float]] = []
        for name in sorted(set(clip_ids)):
            mask = np.array([clip == name for clip in clip_ids])
            named.append((name, float(np.mean(pred[model][mask] == labels[mask]))))
        values = np.array([score for _name, score in named], dtype=np.float64)
        estimate, low, high = bootstrap_mean(values, draws=draws, seed=seed)
        estimates.append(
            ProbeEstimate(
                vae=vae,
                source_codec=source_codec,
                task="classification",
                model=model,
                metric="accuracy",
                clips=int(values.size),
                estimate=estimate,
                ci_low=low,
                ci_high=high,
                ridge_lambda=lam,
                bootstrap_draws=draws,
            )
        )
        for clip_id, score in named:
            scores.append(
                _ClipScore(
                    vae=vae,
                    source_codec=source_codec,
                    clip_id=clip_id,
                    model=model,
                    task="classification",
                    score=score,
                )
            )
    return estimates, scores


def _pool_models(
    scores: list[_ClipScore],
    *,
    vae: str,
    source_codec: str,
    lam: float,
    draws: int,
    seed: int,
) -> list[ProbeEstimate]:
    pooled: list[ProbeEstimate] = []
    for model in _MODEL_ORDER:
        by_clip: dict[str, list[float]] = defaultdict(list)
        for row in scores:
            if row.model != model:
                continue
            by_clip[row.clip_id].append(row.score)
        if len(by_clip) < 2:
            continue
        values = np.array(
            [float(np.mean(by_clip[name])) for name in sorted(by_clip)],
            dtype=np.float64,
        )
        estimate, low, high = bootstrap_mean(values, draws=draws, seed=seed)
        pooled.append(
            ProbeEstimate(
                vae=vae,
                source_codec=source_codec,
                task="classification",
                model=model,
                metric="accuracy",
                clips=int(values.size),
                estimate=estimate,
                ci_low=low,
                ci_high=high,
                ridge_lambda=lam,
                bootstrap_draws=draws,
            )
        )
    return pooled


def _accuracy_section(intro: str, estimates: list[ProbeEstimate]) -> str:
    rows = [row for row in estimates if row.task == "classification"]
    if not rows:
        return ""
    lines = [
        intro,
        "",
        "| VAE | Source | Model | Clips | Accuracy | 95% CI |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        label = _CODEC_LABEL.get(row.source_codec, row.source_codec)
        span = f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        lines.append(
            f"| {row.vae} | {label} | {row.model} | {row.clips} | {row.estimate:.3f} | {span} |"
        )
    return "\n".join(lines) + "\n"


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
