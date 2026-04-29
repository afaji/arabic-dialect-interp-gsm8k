# Geometric Complexity Scaling

This project tests whether task complexity induces increasingly nonlinear residual-stream geometry in `google/gemma-4-E2B-it`.

The suite runs three tasks over 3 random seeds, samples 2,000 rows per task/seed before inference, filters to correct answers, computes geometry metrics on the retained activations, and plots mean/std error bars.

## Tasks

- **Type I, fact retrieval:** `mandarjoshi/trivia_qa`, using `question` and answer aliases.
- **Type II, sentiment:** `SetFit/sst2`, using `text` and `label_text`.
- **Type III, math reasoning:** `openai/gsm8k` with config `main`, using `question` and `answer`.

## Install

Recommended Conda setup:

```bash
conda env create -f environment.yml
conda activate geometric-complexity-scaling
python -m pip install -e .
```

Cluster/local-prefix setup, matching this workspace:

```bash
conda env create -p /scratch/afz225/res_stream_project/.conda-env -f environment.yml
conda activate /scratch/afz225/res_stream_project/.conda-env
python -m pip install -e . --no-build-isolation
```

For pip-only environments:

```bash
python -m pip install -e .
```

You may also need to authenticate with Hugging Face if the Gemma checkpoint requires gated access:

```bash
huggingface-cli login
```

## Run

Small smoke test:

```bash
gcs-run-all --sample-size 5 --max-correct-geometry 5 --max-new-tokens 64 --output-dir outputs/smoke
```

Full planned run:

```bash
gcs-run-all --sample-size 2000 --seeds 0 1 2 --output-dir outputs/gemma4_e2b
```

## HPC Slurm Run

The full HPC pipeline uses a Slurm array over the 9 `(task, seed)` combinations and caps concurrency at 4 GPU jobs:

```bash
bash slurm/submit_pipeline.sh
```

This submits:

- `slurm/run_inference_array.sbatch`: `#SBATCH --array=0-8%4`, one GPU per array task.
- `slurm/run_postprocess.sbatch`: computes metrics and visualizations after the inference array succeeds.

Manual submission:

```bash
INFER_JOB=$(sbatch --parsable slurm/run_inference_array.sbatch)
sbatch --dependency=afterok:${INFER_JOB} slurm/run_postprocess.sbatch
```

Logs are written to `slurm/logs/`; outputs are written to `outputs/gemma4_e2b/`.
Hugging Face caches are read/written under `/scratch/afz225/.cache`.

To backfill incorrect-example activations from existing generations without rerunning generation:

```bash
bash slurm/submit_backfill_pipeline.sh
```

To generate residual-stream trajectory dimensionality-reduction plots with correctness coloring:

```bash
python scripts/plot_trajectory_dr.py --output-dir outputs/gemma4_e2b --max-per-group 75
```

This writes separate PCA and Isomap plots to `outputs/gemma4_e2b/plots/`, with one panel per task.

## Tests

Unit tests do not download Gemma:

```bash
pytest
```

Optional dataset contract tests check real dataset schemas and may require network or a populated Hugging Face cache:

```bash
RUN_DATASET_CONTRACT_TESTS=1 pytest -m integration
```

The Gemma prompt path uses:

```python
processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

Thinking mode is intentionally disabled for every task.

## Outputs

- `inference/*.jsonl`: prompt, target, output, parsed answer, correctness, seed, row id, generated token metadata.
- `activations/*.npz`: correct-example residual-stream arrays for geometry.
- `metrics/*.csv`: per-task/per-seed/per-layer metrics.
- `plots/*.png`: accuracy, retained correct counts, PCA, Isomap, LLE, intrinsic dimension, and curvature plots.

The semantic-pullback curvature is approximated by projecting residual states through a deterministic sampled subset of the model output embedding/unembedding directions. This keeps the experiment tractable while preserving a separate logit-space geometry probe from raw Euclidean residual curvature.

TriviaQA correctness uses official-style normalization, exact match over all answer aliases, and token F1. A fact answer is retained for geometry when exact match succeeds or its best alias F1 is at least 0.5; the raw score details are saved in each inference record.
