import csv
import json

import numpy as np

from geometric_complexity_scaling.cli import custom_math_pca_trajectory_main, custom_math_postprocess_plots_main
from geometric_complexity_scaling.custom_math import configure_cache_dirs, load_gsm8k_like_rows, parse_row_filters
from geometric_complexity_scaling.plotting import (
    infer_custom_math_gallery_prefix,
    plot_custom_pca_trajectory,
    plot_custom_pca_trajectory_comparison,
    plot_custom_pca_trajectory_gallery,
)


def test_load_gsm8k_like_jsonl_with_custom_columns(tmp_path):
    path = tmp_path / "math.jsonl"
    rows = [
        {"problem": "A has 2 and gets 3. How many?", "gold": "#### 5", "id": "a"},
        {"problem": "10 minus 4?", "gold": "6", "id": "b"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = load_gsm8k_like_rows(
        data_file=path,
        question_column="problem",
        answer_column="gold",
        row_id_column="id",
    )

    assert loaded[0]["question"] == rows[0]["problem"]
    assert loaded[0]["answer"] == "#### 5"
    assert loaded[0]["row_id"] == "a"


def test_load_gsm8k_like_csv(tmp_path):
    path = tmp_path / "math.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerow({"question": "2+2?", "answer": "#### 4"})

    loaded = load_gsm8k_like_rows(data_file=path)

    assert loaded == [
        {
            "question": "2+2?",
            "answer": "#### 4",
            "row_id": 0,
            "source_row": {"question": "2+2?", "answer": "#### 4"},
        }
    ]


def test_load_gsm8k_like_rows_applies_exact_row_filters(tmp_path):
    path = tmp_path / "math.jsonl"
    rows = [
        {"question": "2+2?", "answer": "#### 4", "dialect": "EGY"},
        {"question": "3+3?", "answer": "#### 6", "dialect": "MSA"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = load_gsm8k_like_rows(data_file=path, row_filters=[("dialect", "EGY")])

    assert len(loaded) == 1
    assert loaded[0]["question"] == "2+2?"
    assert loaded[0]["source_row"]["dialect"] == "EGY"


def test_parse_row_filters_accepts_column_equals_value():
    assert parse_row_filters(["dialect=EGY", "subset=no_replacement"]) == [
        ("dialect", "EGY"),
        ("subset", "no_replacement"),
    ]


def test_configure_cache_dirs_sets_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    configure_cache_dirs(
        hf_home=tmp_path / "hf",
        datasets_cache=tmp_path / "datasets",
        transformers_cache=tmp_path / "transformers",
        mpl_cache=tmp_path / "mpl",
    )

    assert (tmp_path / "hf").is_dir()
    assert (tmp_path / "datasets").is_dir()
    assert (tmp_path / "transformers").is_dir()
    assert (tmp_path / "mpl").is_dir()


def test_custom_pca_trajectory_plot_synthetic(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    activations = np.random.default_rng(0).normal(size=(12, 4, 8)).astype(np.float32)
    path = activation_dir / "custom_math_seed0_all_activations.npz"
    np.savez_compressed(
        path,
        activations=activations,
        correct=np.array([True] * 6 + [False] * 6),
        task=np.array("custom_math"),
        seed=np.array(0),
    )

    out = plot_custom_pca_trajectory(path, output_dir=tmp_path, max_per_group=3)

    assert out.exists()
    assert out.name == "custom_math_trajectory_pca_correctness.png"


def test_custom_pca_trajectory_comparison_plot_synthetic(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    rng = np.random.default_rng(0)
    all_path = activation_dir / "custom_math_all-replacement_dialect-egy_seed0_all_activations.npz"
    no_path = activation_dir / "custom_math_no-replacement_dialect-egy_seed0_all_activations.npz"
    np.savez_compressed(
        all_path,
        activations=rng.normal(size=(10, 4, 8)).astype(np.float32),
        correct=np.array([True] * 5 + [False] * 5),
        task=np.array("custom_math"),
        seed=np.array(0),
    )
    np.savez_compressed(
        no_path,
        activations=rng.normal(size=(10, 4, 8)).astype(np.float32),
        correct=np.array([True] * 4 + [False] * 6),
        task=np.array("custom_math"),
        seed=np.array(0),
    )

    out = plot_custom_pca_trajectory_comparison(
        activation_specs=[("no replacement", no_path), ("all replacement", all_path)],
        output_dir=tmp_path,
        plot_name="replacement_comparison",
        max_per_group=3,
    )

    assert out.exists()
    assert out.name == "replacement_comparison_trajectory_pca_correctness.png"


def test_custom_pca_trajectory_gallery_writes_plot_suite(tmp_path):
    activation_dir = tmp_path / "activations"
    inference_dir = tmp_path / "inference"
    activation_dir.mkdir()
    inference_dir.mkdir()
    rng = np.random.default_rng(0)
    all_path = activation_dir / "custom_math_all-replacement_dialect-egy_seed0_all_activations.npz"
    no_path = activation_dir / "custom_math_no-replacement_dialect-egy_seed0_all_activations.npz"
    row_ids = np.array([str(idx) for idx in range(10)], dtype=object)
    np.savez_compressed(
        all_path,
        activations=rng.normal(size=(10, 4, 8)).astype(np.float32),
        row_ids=row_ids,
        correct=np.array([True] * 5 + [False] * 5),
        task=np.array("custom_math"),
        seed=np.array(0),
    )
    np.savez_compressed(
        no_path,
        activations=rng.normal(size=(10, 4, 8)).astype(np.float32),
        row_ids=row_ids,
        correct=np.array([True] * 4 + [False] * 6),
        task=np.array("custom_math"),
        seed=np.array(0),
    )
    all_records = [
        {"row_id": str(idx), "parsed_answer": value}
        for idx, value in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    ]
    no_records = [
        {"row_id": str(idx), "parsed_answer": value}
        for idx, value in enumerate(["1", "0", "3", "0", "5", "0", "7", "0", "9", "0"])
    ]
    with (inference_dir / "custom_math_all-replacement_dialect-egy_seed0.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")
    with (inference_dir / "custom_math_no-replacement_dialect-egy_seed0.jsonl").open("w", encoding="utf-8") as handle:
        for record in no_records:
            handle.write(json.dumps(record) + "\n")

    written = plot_custom_pca_trajectory_gallery(
        activation_specs=[("no replacement", no_path), ("all replacement", all_path)],
        output_dir=tmp_path,
        plot_prefix="custom_math_seed0_dialect-egy",
        max_per_group=3,
    )

    assert {path.name for path in written} == {
        "custom_math_no-replacement_dialect-egy_seed0_trajectory_pca_correctness.png",
        "custom_math_all-replacement_dialect-egy_seed0_trajectory_pca_correctness.png",
        "custom_math_seed0_dialect-egy_replacement_comparison_trajectory_pca_correctness.png",
        "custom_math_seed0_dialect-egy_replacement_only_trajectory_pca.png",
        "custom_math_seed0_dialect-egy_correctness_only_trajectory_pca.png",
        "custom_math_seed0_dialect-egy_replacement_agreement_only_trajectory_pca.png",
        "custom_math_seed0_dialect-egy_replacement_disagreement_only_trajectory_pca.png",
    }
    assert all(path.exists() for path in written)


def test_infer_custom_math_gallery_prefix_from_activation_path():
    prefix = infer_custom_math_gallery_prefix(
        "custom_math_all-replacement_dialect-egy_seed0_all_activations.npz"
    )
    assert prefix == "custom_math_seed0_dialect-egy"


def test_custom_math_cli_help(capsys):
    try:
        custom_math_pca_trajectory_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--data-file" in out
    assert "--row-filter" in out
    assert "--hf-token" in out
    assert "--max-new-tokens" in out


def test_custom_math_postprocess_cli_help(capsys):
    try:
        custom_math_postprocess_plots_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--all-replacement-activation" in out
    assert "--no-replacement-activation" in out
    assert "--plot-prefix" in out
