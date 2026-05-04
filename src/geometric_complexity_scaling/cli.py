from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import aggregate_inference, combine_metric_csvs
from .backfill import backfill_incorrect_activations_for_task
from .custom_math import parse_row_filters, run_custom_math_pca_trajectory
from .geometry import compute_metrics_for_activation_file
from .inference import run_inference_for_task
from .plotting import plot_custom_pca_trajectory_gallery, plot_results, plot_trajectory_dr
from .tasks import TASKS


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="outputs/gemma4_e2b")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--tasks", nargs="+", default=["fact", "sentiment", "math"], choices=sorted(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--max-correct-geometry", type=int, default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true")


def run_inference_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Gemma inference and save correct-example activations.")
    add_common_args(parser)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--geometry-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--semantic-vocab-sample", type=int, default=256)
    args = parser.parse_args(argv)

    for task in args.tasks:
        for seed in args.seeds:
            run_inference_for_task(
                task=task,
                seed=seed,
                output_dir=args.output_dir,
                model_id=args.model_id,
                sample_size=args.sample_size,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                device_map=args.device_map,
                geometry_dtype=args.geometry_dtype,
                max_correct_geometry=args.max_correct_geometry,
                semantic_vocab_sample=args.semantic_vocab_sample,
                overwrite=args.overwrite,
            )
    aggregate_inference(args.output_dir)


def compute_metrics_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute geometry metrics from saved activation files.")
    parser.add_argument("--output-dir", default="outputs/gemma4_e2b")
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--semantic-projection-dim", type=int, default=256)
    args = parser.parse_args(argv)

    metrics_dir = Path(args.output_dir) / "metrics"
    if metrics_dir.exists():
        for path in metrics_dir.glob("*_metrics.csv"):
            path.unlink()
    for path in sorted((Path(args.output_dir) / "activations").glob("*_activations.npz")):
        compute_metrics_for_activation_file(
            activation_path=path,
            output_dir=args.output_dir,
            pca_components=args.pca_components,
            neighbors=args.neighbors,
            semantic_projection_dim=args.semantic_projection_dim,
        )
    combine_metric_csvs(args.output_dir)


def backfill_incorrect_activations_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill incorrect-example activations from saved generations.")
    add_common_args(parser)
    parser.add_argument("--geometry-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--semantic-vocab-sample", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)

    for task in args.tasks:
        for seed in args.seeds:
            backfill_incorrect_activations_for_task(
                task=task,
                seed=seed,
                output_dir=args.output_dir,
                model_id=args.model_id,
                dtype=args.dtype,
                device_map=args.device_map,
                geometry_dtype=args.geometry_dtype,
                semantic_vocab_sample=args.semantic_vocab_sample,
                max_examples=args.max_examples,
                overwrite=args.overwrite,
            )


def plot_results_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot inference and geometry summaries.")
    parser.add_argument("--output-dir", default="outputs/gemma4_e2b")
    args = parser.parse_args(argv)
    plot_results(args.output_dir)


def plot_trajectory_dr_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot PCA/Isomap residual-stream trajectories with correctness coloring.")
    parser.add_argument("--output-dir", default="outputs/gemma4_e2b")
    parser.add_argument("--max-per-group", type=int, default=75)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--isomap-neighbors", type=int, default=12)
    parser.add_argument("--no-layer-normalize", action="store_true")
    args = parser.parse_args(argv)
    plot_trajectory_dr(
        output_dir=args.output_dir,
        max_per_group=args.max_per_group,
        random_seed=args.random_seed,
        isomap_neighbors=args.isomap_neighbors,
        normalize_layers=not args.no_layer_normalize,
    )


def custom_math_pca_trajectory_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a model on GSM8K-like data and plot PCA residual-stream trajectories colored by correctness."
    )
    parser.add_argument("--model-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-file", default=None)
    source.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--row-id-column", default=None)
    parser.add_argument("--row-filter", action="append", default=[], help="Exact-match filter in COLUMN=VALUE form.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="outputs/custom_math")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--geometry-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max-per-group", type=int, default=75)
    parser.add_argument("--no-layer-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--datasets-cache", default=None)
    parser.add_argument("--transformers-cache", default=None)
    parser.add_argument("--mpl-cache", default=None)
    args = parser.parse_args(argv)
    run_custom_math_pca_trajectory(
        output_dir=args.output_dir,
        model_id=args.model_id,
        data_file=args.data_file,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        question_column=args.question_column,
        answer_column=args.answer_column,
        row_id_column=args.row_id_column,
        row_filters=parse_row_filters(args.row_filter),
        sample_size=args.sample_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
        geometry_dtype=args.geometry_dtype,
        max_per_group=args.max_per_group,
        normalize_layers=not args.no_layer_normalize,
        overwrite=args.overwrite,
        hf_home=args.hf_home,
        hf_token=args.hf_token,
        datasets_cache=args.datasets_cache,
        transformers_cache=args.transformers_cache,
        mpl_cache=args.mpl_cache,
        run_name=args.run_name,
    )


def custom_math_postprocess_plots_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate custom math PCA plot galleries from saved activation files without rerunning inference."
    )
    parser.add_argument("--all-replacement-activation", required=True)
    parser.add_argument("--no-replacement-activation", required=True)
    parser.add_argument("--output-dir", default="outputs/custom_math")
    parser.add_argument("--plot-prefix", default=None)
    parser.add_argument("--max-per-group", type=int, default=75, help="Per-label cap. Use 0 to keep all examples.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--no-layer-normalize", action="store_true")
    args = parser.parse_args(argv)
    plot_custom_pca_trajectory_gallery(
        activation_specs=[
            ("no replacement", args.no_replacement_activation),
            ("all replacement", args.all_replacement_activation),
        ],
        output_dir=args.output_dir,
        plot_prefix=args.plot_prefix,
        max_per_group=args.max_per_group,
        random_seed=args.random_seed,
        normalize_layers=not args.no_layer_normalize,
    )


def run_all_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference, geometry metrics, and plotting.")
    add_common_args(parser)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--geometry-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--semantic-vocab-sample", type=int, default=256)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--semantic-projection-dim", type=int, default=256)
    args = parser.parse_args(argv)

    for task in args.tasks:
        for seed in args.seeds:
            run_inference_for_task(
                task=task,
                seed=seed,
                output_dir=args.output_dir,
                model_id=args.model_id,
                sample_size=args.sample_size,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                device_map=args.device_map,
                geometry_dtype=args.geometry_dtype,
                max_correct_geometry=args.max_correct_geometry,
                semantic_vocab_sample=args.semantic_vocab_sample,
                overwrite=args.overwrite,
            )
    aggregate_inference(args.output_dir)
    for path in sorted((Path(args.output_dir) / "activations").glob("*_activations.npz")):
        compute_metrics_for_activation_file(
            activation_path=path,
            output_dir=args.output_dir,
                pca_components=args.pca_components,
                neighbors=args.neighbors,
                semantic_projection_dim=args.semantic_projection_dim,
            )
    combine_metric_csvs(args.output_dir)
    plot_results(args.output_dir)
