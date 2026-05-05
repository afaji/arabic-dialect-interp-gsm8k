from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np

from .scoring import score_answer
from .utils import ensure_dir, require_import

REPLACEMENT_CORRECTNESS_COLORS = {
    "no replacement correct": "#8b1e1e",
    "no replacement incorrect": "#e88f8f",
    "all replacement correct": "#1f7a3a",
    "all replacement incorrect": "#95d5a6",
}
REPLACEMENT_CORRECTNESS_LINE_STYLES = {
    "no replacement correct": "-",
    "no replacement incorrect": "--",
    "all replacement correct": "-",
    "all replacement incorrect": "--",
}
SUBSET_ONLY_COLORS = {"no replacement": "#8b1e1e", "all replacement": "#1f7a3a"}
CORRECTNESS_ONLY_COLORS = {"correct": "#14833b", "incorrect": "#c9342f"}
CORRECTNESS_LINE_STYLES = {"correct": "-", "incorrect": "--"}
REPLACEMENT_ACTIVATION_STEM_RE = re.compile(
    r"^custom_math_(?P<subset>all-replacement|no-replacement)(?P<suffix>.*)_seed(?P<seed>\d+)_all_activations$"
)


def plot_results(output_dir: str | Path) -> list[Path]:
    pd = require_import("pandas", "pandas")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    sns = require_import("seaborn", "seaborn")
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "plots")
    written: list[Path] = []

    inference_path = output_dir / "metrics" / "inference_summary.csv"
    if inference_path.exists():
        summary = pd.read_csv(inference_path)
        written.append(_barplot(summary, "accuracy", "Accuracy", plot_dir / "accuracy_by_task.png", sns, plt))
        written.append(
            _barplot(summary, "num_correct", "Correct examples retained", plot_dir / "correct_counts_by_task.png", sns, plt)
        )

    metrics_path = output_dir / "metrics" / "geometry_metrics_all.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for metric_name in [
            "pca_residual_variance",
            "pca_reconstruction_mse",
            "isomap_reconstruction_error",
            "lle_reconstruction_error",
            "intrinsic_dim_twonn",
            "local_id_mle_k10",
            "local_id_mle_k20",
            "euclidean_layer_curvature",
            "semantic_pullback_layer_curvature",
        ]:
            subset = metrics[metrics["metric_name"] == metric_name].dropna(subset=["metric_value"])
            if not subset.empty and "layer" in subset:
                out = plot_dir / f"{metric_name}_by_layer.png"
                _lineplot(subset, metric_name, out, sns, plt)
                written.append(out)
    return written


def plot_trajectory_dr(
    output_dir: str | Path,
    max_per_group: int = 75,
    random_seed: int = 0,
    isomap_neighbors: int = 12,
    normalize_layers: bool = True,
) -> list[Path]:
    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    manifold = require_import("sklearn.manifold", "scikit-learn")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "plots")
    tasks = ["fact", "math", "sentiment"]
    written = []
    for method in ("pca", "isomap"):
        out = plot_dir / f"trajectory_{method}_correctness.png"
        _plot_trajectory_method(
            output_dir=output_dir,
            output_path=out,
            tasks=tasks,
            method=method,
            PCA=decomposition.PCA,
            Isomap=manifold.Isomap,
            plt=plt,
            max_per_group=max_per_group,
            random_seed=random_seed,
            isomap_neighbors=isomap_neighbors,
            normalize_layers=normalize_layers,
        )
        written.append(out)
    return written


def plot_custom_pca_trajectory(
    activation_path: str | Path,
    output_dir: str | Path,
    plot_name: str = "custom_math",
    max_per_group: int = 75,
    random_seed: int = 0,
    normalize_layers: bool = True,
) -> Path:
    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    payload = np.load(activation_path, allow_pickle=True)
    activations = payload["activations"].astype(np.float32)
    labels = _labels_from_activation_payload(payload)
    trajectories, labels = _subsample_trajectories_by_label(activations, labels, max_per_group, random_seed)
    plot_dir = ensure_dir(output_dir / "plots")
    output_path = plot_dir / f"{plot_name}_trajectory_pca_correctness.png"
    _plot_single_pca_trajectory(
        trajectories=trajectories,
        labels=labels,
        output_path=output_path,
        title=f"{plot_name} residual-stream layer trajectories (PCA, layer-normalized)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
    )
    return output_path


def plot_custom_pca_trajectory_gallery(
    activation_specs: list[tuple[str, str | Path]],
    output_dir: str | Path,
    plot_prefix: str | None = None,
    max_per_group: int = 75,
    random_seed: int = 0,
    normalize_layers: bool = True,
) -> list[Path]:
    if len(activation_specs) < 2:
        raise ValueError("Need at least two activation specs to build the comparison gallery.")

    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    full_payloads = [_load_named_custom_activation(name, activation_path) for name, activation_path in activation_specs]
    sampled_payloads = [
        _sample_named_custom_activation(
            payload,
            max_per_group=max_per_group,
            random_seed=random_seed,
        )
        for payload in full_payloads
    ]
    shared_fit_trajectories = _concatenate_trajectories([payload["activations"] for payload in sampled_payloads])
    plot_prefix = plot_prefix or _infer_custom_math_gallery_prefix(activation_specs)
    plot_dir = ensure_dir(output_dir / "plots")
    written: list[Path] = []

    for payload in sampled_payloads:
        subset_colors = _subset_correctness_colors(payload["name"])
        local_stem = _activation_plot_stem(payload["path"])
        local_path = plot_dir / f"{local_stem}_trajectory_pca_correctness.png"
        _plot_single_pca_trajectory(
            trajectories=payload["activations"],
            labels=payload["correctness_labels"],
            output_path=local_path,
            title=f"{payload['name']} residual-stream trajectories (PCA, local basis)",
            PCA=decomposition.PCA,
            plt=plt,
            random_seed=random_seed,
            normalize_layers=normalize_layers,
            colors=subset_colors,
            legend_order=["correct", "incorrect"],
            line_styles=CORRECTNESS_LINE_STYLES,
        )
        written.append(local_path)
        (plot_dir / f"{local_stem}_shared-basis_trajectory_pca_correctness.png").unlink(missing_ok=True)

    four_group_path = plot_dir / f"{plot_prefix}_replacement_comparison_trajectory_pca_correctness.png"
    four_group_trajectories, four_group_labels = _combine_named_payloads(sampled_payloads, label_mode="subset_and_correctness")
    _plot_single_pca_trajectory(
        trajectories=four_group_trajectories,
        labels=four_group_labels,
        output_path=four_group_path,
        title="Arabic GSM8K residual-stream trajectories (PCA, shared basis)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
        colors=REPLACEMENT_CORRECTNESS_COLORS,
        legend_order=[
            "no replacement correct",
            "no replacement incorrect",
            "all replacement correct",
            "all replacement incorrect",
        ],
        line_styles=REPLACEMENT_CORRECTNESS_LINE_STYLES,
        legend_ncol=3,
        fit_trajectories=shared_fit_trajectories,
    )
    written.append(four_group_path)

    subset_only_path = plot_dir / f"{plot_prefix}_replacement_only_trajectory_pca.png"
    subset_only_trajectories, subset_only_labels = _combine_named_payloads(sampled_payloads, label_mode="subset")
    _plot_single_pca_trajectory(
        trajectories=subset_only_trajectories,
        labels=subset_only_labels,
        output_path=subset_only_path,
        title="Arabic GSM8K residual-stream trajectories by replacement subset (PCA, shared basis)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
        colors=SUBSET_ONLY_COLORS,
        legend_order=["no replacement", "all replacement"],
        fit_trajectories=shared_fit_trajectories,
    )
    written.append(subset_only_path)

    correctness_only_path = plot_dir / f"{plot_prefix}_correctness_only_trajectory_pca.png"
    correctness_only_trajectories, correctness_only_labels = _combine_named_payloads(
        sampled_payloads,
        label_mode="correctness",
    )
    _plot_single_pca_trajectory(
        trajectories=correctness_only_trajectories,
        labels=correctness_only_labels,
        output_path=correctness_only_path,
        title="Arabic GSM8K residual-stream trajectories by correctness (PCA, shared basis)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
        colors=CORRECTNESS_ONLY_COLORS,
        legend_order=["correct", "incorrect"],
        line_styles=CORRECTNESS_LINE_STYLES,
        fit_trajectories=shared_fit_trajectories,
    )
    written.append(correctness_only_path)

    correctness_flip_payloads = _split_named_payloads_by_correctness_inconsistency(
        full_payloads=full_payloads,
        max_per_group=max_per_group,
        random_seed=random_seed,
    )
    correctness_flip_path = plot_dir / f"{plot_prefix}_replacement_correctness_inconsistent_only_trajectory_pca_correctness.png"
    correctness_flip_trajectories, correctness_flip_labels = _combine_named_payloads(
        correctness_flip_payloads,
        label_mode="subset_and_correctness",
    )
    correctness_flip_fit_trajectories = _concatenate_trajectories(
        [payload["activations"] for payload in correctness_flip_payloads]
    )
    _plot_single_pca_trajectory(
        trajectories=correctness_flip_trajectories,
        labels=correctness_flip_labels,
        output_path=correctness_flip_path,
        title="Arabic GSM8K trajectories where correctness flips between settings (PCA, shared basis)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
        colors=REPLACEMENT_CORRECTNESS_COLORS,
        legend_order=[
            "all replacement correct",
            "no replacement incorrect",
            "no replacement correct",
            "all replacement incorrect",
        ],
        line_styles=REPLACEMENT_CORRECTNESS_LINE_STYLES,
        legend_ncol=3,
        fit_trajectories=correctness_flip_fit_trajectories,
    )
    written.append(correctness_flip_path)

    agreement_payloads, disagreement_payloads = _split_named_payloads_by_output_agreement(
        full_payloads=full_payloads,
        max_per_group=max_per_group,
        random_seed=random_seed,
    )
    for suffix, title, filtered_payloads in [
        (
            "replacement_agreement_only",
            "Arabic GSM8K trajectories where both settings give the same final answer (PCA, shared basis)",
            agreement_payloads,
        ),
        (
            "replacement_disagreement_only",
            "Arabic GSM8K trajectories where the settings give different final answers (PCA, shared basis)",
            disagreement_payloads,
        ),
    ]:
        pair_path = plot_dir / f"{plot_prefix}_{suffix}_trajectory_pca.png"
        pair_trajectories, pair_labels = _combine_named_payloads(filtered_payloads, label_mode="subset")
        pair_fit_trajectories = _concatenate_trajectories([payload["activations"] for payload in filtered_payloads])
        _plot_single_pca_trajectory(
            trajectories=pair_trajectories,
            labels=pair_labels,
            output_path=pair_path,
            title=title,
            PCA=decomposition.PCA,
            plt=plt,
            random_seed=random_seed,
            normalize_layers=normalize_layers,
            colors=SUBSET_ONLY_COLORS,
            legend_order=["no replacement", "all replacement"],
            fit_trajectories=pair_fit_trajectories,
        )
        written.append(pair_path)
    return written


def plot_custom_pca_trajectory_comparison(
    activation_specs: list[tuple[str, str | Path]],
    output_dir: str | Path,
    plot_name: str = "custom_math_comparison",
    max_per_group: int = 75,
    random_seed: int = 0,
    normalize_layers: bool = True,
) -> Path:
    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    sampled_payloads = [
        _sample_named_custom_activation(
            _load_named_custom_activation(prefix, activation_path),
            max_per_group=max_per_group,
            random_seed=random_seed,
        )
        for prefix, activation_path in activation_specs
    ]
    trajectories, combined_labels = _combine_named_payloads(sampled_payloads, label_mode="subset_and_correctness")
    shared_fit_trajectories = _concatenate_trajectories([payload["activations"] for payload in sampled_payloads])

    plot_dir = ensure_dir(output_dir / "plots")
    output_path = plot_dir / f"{plot_name}_trajectory_pca_correctness.png"
    _plot_single_pca_trajectory(
        trajectories=trajectories,
        labels=combined_labels,
        output_path=output_path,
        title="Arabic GSM8K residual-stream layer trajectories (PCA, shared basis)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
        colors=REPLACEMENT_CORRECTNESS_COLORS,
        legend_order=[
            "no replacement correct",
            "no replacement incorrect",
            "all replacement correct",
            "all replacement incorrect",
        ],
        line_styles=REPLACEMENT_CORRECTNESS_LINE_STYLES,
        legend_ncol=3,
        fit_trajectories=shared_fit_trajectories,
    )
    return output_path


def _barplot(df, y_col, title, output_path, sns, plt):
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="task", y=y_col, errorbar="sd")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _lineplot(df, title, output_path, sns, plt):
    plt.figure(figsize=(9, 5))
    hue = "task" if "outcome_group" not in df.columns else "task"
    style = "outcome_group" if "outcome_group" in df.columns else None
    sns.lineplot(
        data=df,
        x="layer",
        y="metric_value",
        hue=hue,
        style=style,
        errorbar="sd",
        estimator="mean",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_trajectory_method(
    output_dir,
    output_path,
    tasks,
    method,
    PCA,
    Isomap,
    plt,
    max_per_group,
    random_seed,
    isomap_neighbors,
    normalize_layers,
):
    fig, axes = plt.subplots(1, len(tasks), figsize=(18, 5), squeeze=False)
    rng = np.random.default_rng(random_seed)
    colors = {"correct": "#14833b", "incorrect": "#c9342f"}
    for axis, task in zip(axes[0], tasks):
        trajectories, labels = _load_subsampled_task_trajectories(
            output_dir=output_dir,
            task=task,
            max_per_group=max_per_group,
            rng=rng,
        )
        if trajectories.size == 0:
            axis.set_title(f"{task}: no activations")
            axis.axis("off")
            continue
        if normalize_layers:
            trajectories = _layer_normalize(trajectories)
        n_examples, n_layers, hidden_dim = trajectories.shape
        states = trajectories.reshape(n_examples * n_layers, hidden_dim)
        if method == "pca":
            embedding = PCA(n_components=2, random_state=random_seed).fit_transform(states)
        elif method == "isomap":
            neighbors = max(2, min(isomap_neighbors, states.shape[0] - 1))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold._isomap")
                warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse._index")
                embedding = Isomap(n_neighbors=neighbors, n_components=2).fit_transform(states)
        else:
            raise ValueError(f"Unknown trajectory DR method: {method}")
        projected = embedding.reshape(n_examples, n_layers, 2)
        _draw_trajectories(axis, projected, labels, colors)
        axis.set_title(task)
        axis.set_xlabel(f"{method.upper()} 1")
        axis.set_ylabel(f"{method.upper()} 2")
    handles = [
        plt.Line2D([0], [0], color=colors["correct"], lw=2, label="correct"),
        plt.Line2D([0], [0], color=colors["incorrect"], lw=2, label="incorrect"),
        plt.Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label="start"),
        plt.Line2D([0], [0], marker="s", color="black", lw=0, markersize=5, label="end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    title = f"Residual-stream layer trajectories ({method.upper()}, layer-normalized)"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_single_pca_trajectory(
    trajectories,
    labels,
    output_path,
    title,
    PCA,
    plt,
    random_seed,
    normalize_layers,
    colors=None,
    legend_order=None,
    line_styles=None,
    legend_ncol=None,
    fit_trajectories=None,
):
    fig, axis = plt.subplots(1, 1, figsize=(7, 6))
    colors = colors or {"correct": "#14833b", "incorrect": "#c9342f"}
    legend_order = legend_order or ["correct", "incorrect"]
    line_styles = line_styles or {}
    if trajectories.size == 0:
        axis.set_title("No activations")
        axis.axis("off")
    else:
        projected = _project_trajectories_with_pca(
            trajectories=trajectories,
            PCA=PCA,
            random_seed=random_seed,
            normalize_layers=normalize_layers,
            fit_trajectories=fit_trajectories,
        )
        _draw_trajectories(axis, projected, labels, colors, line_styles)
        axis.set_title(title)
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
    handles = [
        plt.Line2D([0], [0], color=colors[label], linestyle=line_styles.get(label, "-"), lw=2, label=label)
        for label in legend_order
        if label in colors
    ] + [
        plt.Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label="start"),
        plt.Line2D([0], [0], marker="s", color="black", lw=0, markersize=5, label="end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=legend_ncol or min(len(handles), 6), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _load_subsampled_task_trajectories(output_dir, task, max_per_group, rng):
    activation_dir = Path(output_dir) / "activations"
    trajectories = []
    labels = []
    for outcome_group in ("correct", "incorrect"):
        group_arrays = []
        for path in sorted(activation_dir.glob(f"{task}_seed*_{outcome_group}_activations.npz")):
            payload = np.load(path, allow_pickle=True)
            activations = payload["activations"].astype(np.float32)
            if activations.ndim == 3 and activations.shape[0] > 0:
                group_arrays.append(activations)
        if not group_arrays:
            continue
        group = np.concatenate(group_arrays, axis=0)
        count = min(max_per_group, group.shape[0])
        selected = rng.choice(group.shape[0], size=count, replace=False)
        trajectories.append(group[selected])
        labels.extend([outcome_group] * count)
    if not trajectories:
        return np.empty((0, 0, 0), dtype=np.float32), []
    return np.concatenate(trajectories, axis=0), labels


def _labels_from_activation_payload(payload) -> list[str]:
    if "outcome_group" in payload:
        values = payload["outcome_group"]
        if np.ndim(values) == 0:
            return [str(values)] * int(payload["activations"].shape[0])
        return [str(value) for value in values.tolist()]
    if "correct" in payload:
        return ["correct" if bool(value) else "incorrect" for value in payload["correct"].tolist()]
    return ["unknown"] * int(payload["activations"].shape[0])


def _subsample_trajectories_by_label(activations, labels, max_per_group, random_seed):
    return _subsample_trajectories_by_explicit_labels(
        activations,
        labels,
        ["correct", "incorrect"],
        max_per_group=max_per_group,
        random_seed=random_seed,
    )


def _subsample_trajectories_by_explicit_labels(activations, labels, label_order, max_per_group, random_seed):
    if activations.ndim != 3 or activations.shape[0] == 0:
        return np.empty((0, 0, 0), dtype=np.float32), []
    rng = np.random.default_rng(random_seed)
    selected_arrays = []
    selected_labels = []
    label_array = np.array(labels)
    for label in label_order:
        indices = np.flatnonzero(label_array == label)
        if indices.size == 0:
            continue
        if max_per_group is None or max_per_group <= 0 or max_per_group >= indices.size:
            chosen = indices
        else:
            chosen = rng.choice(indices, size=max_per_group, replace=False)
        selected_arrays.append(activations[chosen])
        selected_labels.extend([label] * int(chosen.size))
    if not selected_arrays:
        return np.empty((0, 0, 0), dtype=np.float32), []
    return np.concatenate(selected_arrays, axis=0), selected_labels


def _layer_normalize(trajectories):
    means, stds = _layer_normalization_stats(trajectories)
    return _apply_layer_normalization(trajectories, means, stds)


def infer_custom_math_gallery_prefix(activation_path: str | Path) -> str:
    match = REPLACEMENT_ACTIVATION_STEM_RE.match(Path(activation_path).stem)
    if not match:
        return "custom_math"
    return f"custom_math_seed{match.group('seed')}{match.group('suffix')}"


def _activation_plot_stem(activation_path: str | Path) -> str:
    stem = Path(activation_path).stem
    if stem.endswith("_all_activations"):
        return stem.removesuffix("_all_activations")
    return stem


def _subset_correctness_colors(subset_name: str) -> dict[str, str]:
    if subset_name == "no replacement":
        return {
            "correct": REPLACEMENT_CORRECTNESS_COLORS["no replacement correct"],
            "incorrect": REPLACEMENT_CORRECTNESS_COLORS["no replacement incorrect"],
        }
    if subset_name == "all replacement":
        return {
            "correct": REPLACEMENT_CORRECTNESS_COLORS["all replacement correct"],
            "incorrect": REPLACEMENT_CORRECTNESS_COLORS["all replacement incorrect"],
        }
    return CORRECTNESS_ONLY_COLORS


def _load_named_custom_activation(name: str, activation_path: str | Path) -> dict[str, object]:
    path = Path(activation_path)
    payload = np.load(path, allow_pickle=True)
    activations = payload["activations"].astype(np.float32)
    correctness_labels = _labels_from_activation_payload(payload)
    row_ids = (
        [str(value) for value in payload["row_ids"].tolist()]
        if "row_ids" in payload
        else [str(idx) for idx in range(int(activations.shape[0]))]
    )
    return {
        "name": name,
        "path": path,
        "activations": activations,
        "correctness_labels": correctness_labels,
        "row_ids": row_ids,
        "correctness_by_row_id": {row_id: label for row_id, label in zip(row_ids, correctness_labels, strict=False)},
        "parsed_answers_by_row_id": _load_parsed_answers_for_activation_path(path),
    }


def _sample_named_custom_activation(payload: dict[str, object], max_per_group: int, random_seed: int) -> dict[str, object]:
    activations, labels = _subsample_trajectories_by_explicit_labels(
        payload["activations"],
        payload["correctness_labels"],
        ["correct", "incorrect"],
        max_per_group=max_per_group,
        random_seed=random_seed,
    )
    return {
        "name": payload["name"],
        "path": payload["path"],
        "activations": activations,
        "correctness_labels": labels,
    }


def _filter_named_custom_activation_by_row_ids(
    payload: dict[str, object],
    row_ids_to_keep: set[str],
) -> dict[str, object]:
    row_ids = payload.get("row_ids", [])
    activations = payload["activations"]
    labels = payload["correctness_labels"]
    if not row_ids or activations.ndim != 3 or activations.shape[0] == 0:
        return {
            "name": payload["name"],
            "path": payload["path"],
            "activations": np.empty((0, 0, 0), dtype=np.float32),
            "correctness_labels": [],
            "row_ids": [],
            "parsed_answers_by_row_id": payload.get("parsed_answers_by_row_id", {}),
        }
    keep_indices = [idx for idx, row_id in enumerate(row_ids) if row_id in row_ids_to_keep]
    return {
        "name": payload["name"],
        "path": payload["path"],
        "activations": activations[keep_indices] if keep_indices else np.empty((0, 0, 0), dtype=np.float32),
        "correctness_labels": [labels[idx] for idx in keep_indices],
        "row_ids": [row_ids[idx] for idx in keep_indices],
        "parsed_answers_by_row_id": payload.get("parsed_answers_by_row_id", {}),
    }


def _split_named_payloads_by_output_agreement(
    full_payloads: list[dict[str, object]],
    max_per_group: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(full_payloads) < 2:
        return [], []
    shared_row_ids = set(full_payloads[0].get("row_ids", []))
    for payload in full_payloads[1:]:
        shared_row_ids &= set(payload.get("row_ids", []))
    agree_row_ids: set[str] = set()
    disagree_row_ids: set[str] = set()
    payload_a = full_payloads[0]
    payload_b = full_payloads[1]
    parsed_a = payload_a.get("parsed_answers_by_row_id", {})
    parsed_b = payload_b.get("parsed_answers_by_row_id", {})
    for row_id in shared_row_ids:
        if _math_outputs_agree(parsed_a.get(row_id), parsed_b.get(row_id)):
            agree_row_ids.add(row_id)
        else:
            disagree_row_ids.add(row_id)
    agreement_payloads = [
        _sample_named_custom_activation(
            _filter_named_custom_activation_by_row_ids(payload, agree_row_ids),
            max_per_group=max_per_group,
            random_seed=random_seed,
        )
        for payload in full_payloads
    ]
    disagreement_payloads = [
        _sample_named_custom_activation(
            _filter_named_custom_activation_by_row_ids(payload, disagree_row_ids),
            max_per_group=max_per_group,
            random_seed=random_seed,
        )
        for payload in full_payloads
    ]
    return agreement_payloads, disagreement_payloads


def _split_named_payloads_by_correctness_inconsistency(
    full_payloads: list[dict[str, object]],
    max_per_group: int,
    random_seed: int,
) -> list[dict[str, object]]:
    if len(full_payloads) < 2:
        return []
    shared_row_ids = set(full_payloads[0].get("row_ids", []))
    for payload in full_payloads[1:]:
        shared_row_ids &= set(payload.get("row_ids", []))
    correctness_maps = [payload.get("correctness_by_row_id", {}) for payload in full_payloads]
    inconsistent_row_ids = {
        row_id
        for row_id in shared_row_ids
        if len({correctness_map.get(row_id) for correctness_map in correctness_maps}) > 1
    }
    return [
        _sample_named_custom_activation(
            _filter_named_custom_activation_by_row_ids(payload, inconsistent_row_ids),
            max_per_group=max_per_group,
            random_seed=random_seed,
        )
        for payload in full_payloads
    ]


def _load_parsed_answers_for_activation_path(activation_path: str | Path) -> dict[str, object]:
    activation_path = Path(activation_path)
    inference_dir = activation_path.parent.parent / "inference"
    inference_path = inference_dir / f"{_activation_plot_stem(activation_path)}.jsonl"
    if not inference_path.exists():
        return {}
    parsed_answers: dict[str, object] = {}
    with inference_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            parsed_answers[str(record.get("row_id"))] = record.get("parsed_answer")
    return parsed_answers


def _math_outputs_agree(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    return bool(score_answer("math", str(left), str(right)).get("correct"))


def _combine_named_payloads(sampled_payloads: list[dict[str, object]], label_mode: str) -> tuple[np.ndarray, list[str]]:
    combined_arrays = []
    combined_labels: list[str] = []
    for payload in sampled_payloads:
        activations = payload["activations"]
        labels = payload["correctness_labels"]
        if activations.ndim != 3 or activations.shape[0] == 0:
            continue
        combined_arrays.append(activations)
        if label_mode == "subset_and_correctness":
            combined_labels.extend(f"{payload['name']} {label}" for label in labels)
        elif label_mode == "subset":
            combined_labels.extend([str(payload["name"])] * activations.shape[0])
        elif label_mode == "correctness":
            combined_labels.extend(str(label) for label in labels)
        else:
            raise ValueError(f"Unknown label mode: {label_mode}")
    return _concatenate_trajectories(combined_arrays), combined_labels


def _concatenate_trajectories(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [array for array in arrays if array.ndim == 3 and array.shape[0] > 0]
    if not nonempty:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.concatenate(nonempty, axis=0)


def _infer_custom_math_gallery_prefix(activation_specs: list[tuple[str, str | Path]]) -> str:
    for _, activation_path in activation_specs:
        inferred = infer_custom_math_gallery_prefix(activation_path)
        if inferred != "custom_math":
            return inferred
    return "custom_math_comparison"


def _layer_normalization_stats(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if trajectories.ndim != 3 or trajectories.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    normalized = trajectories.astype(np.float32, copy=False)
    means = normalized.mean(axis=0)
    stds = normalized.std(axis=0)
    return means, np.maximum(stds, 1e-6)


def _apply_layer_normalization(trajectories: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    normalized = trajectories.astype(np.float32, copy=True)
    if means.size == 0 or stds.size == 0:
        return normalized
    return (normalized - means) / stds


def _project_trajectories_with_pca(
    trajectories: np.ndarray,
    PCA,
    random_seed: int,
    normalize_layers: bool,
    fit_trajectories: np.ndarray | None = None,
) -> np.ndarray:
    if trajectories.ndim != 3 or trajectories.shape[0] == 0:
        return np.empty((0, 0, 2), dtype=np.float32)
    fit_trajectories = fit_trajectories if fit_trajectories is not None and fit_trajectories.size else trajectories
    plot_trajectories = trajectories.astype(np.float32, copy=False)
    pca_fit_trajectories = fit_trajectories.astype(np.float32, copy=False)
    if normalize_layers:
        means, stds = _layer_normalization_stats(pca_fit_trajectories)
        plot_trajectories = _apply_layer_normalization(plot_trajectories, means, stds)
        pca_fit_trajectories = _apply_layer_normalization(pca_fit_trajectories, means, stds)
    n_examples, n_layers, hidden_dim = plot_trajectories.shape
    fit_states = pca_fit_trajectories.reshape(pca_fit_trajectories.shape[0] * pca_fit_trajectories.shape[1], hidden_dim)
    plot_states = plot_trajectories.reshape(n_examples * n_layers, hidden_dim)
    pca = PCA(n_components=2, random_state=random_seed)
    pca.fit(fit_states)
    embedding = pca.transform(plot_states)
    return embedding.reshape(n_examples, n_layers, 2)


def _draw_trajectories(axis, projected, labels, colors, line_styles=None):
    line_styles = line_styles or {}
    n_layers = projected.shape[1]
    alphas = np.linspace(0.25, 0.95, max(n_layers - 1, 1))
    for trajectory, label in zip(projected, labels):
        color = colors[label]
        linestyle = line_styles.get(label, "-")
        for layer_idx in range(n_layers - 1):
            axis.plot(
                trajectory[layer_idx : layer_idx + 2, 0],
                trajectory[layer_idx : layer_idx + 2, 1],
                color=color,
                linestyle=linestyle,
                alpha=float(alphas[layer_idx]),
                linewidth=0.8,
            )
        axis.scatter(trajectory[0, 0], trajectory[0, 1], color=color, marker="o", s=12, alpha=0.9)
        axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, marker="s", s=14, alpha=0.9)
    axis.grid(alpha=0.2, linewidth=0.5)
