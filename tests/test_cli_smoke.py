from pathlib import Path

from geometric_complexity_scaling.cli import compute_metrics_main, plot_results_main
from tests.test_geometry import write_activation_file


def test_cli_metrics_and_plot_smoke(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    write_activation_file(activation_dir / "fact_seed0_correct_activations.npz")

    compute_metrics_main(["--output-dir", str(tmp_path), "--pca-components", "2", "--neighbors", "3"])
    assert (tmp_path / "metrics" / "geometry_metrics_all.csv").exists()
    plot_results_main(["--output-dir", str(tmp_path)])
    assert (tmp_path / "plots").exists()


def test_console_script_files_exist():
    assert Path("scripts/run_inference.py").exists()
    assert Path("scripts/compute_metrics.py").exists()
    assert Path("scripts/plot_results.py").exists()
    assert Path("scripts/plot_custom_math_postprocess.py").exists()
    assert Path("scripts/run_arabic_gsm8k_egy_pair.py").exists()
