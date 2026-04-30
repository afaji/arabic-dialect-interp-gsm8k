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

## Workflow: GSM8K-Like Data To PCA Trajectory Plot

Use this path when you have a model and a GSM8K-like math dataset and want a residual-stream PCA trajectory plot colored by correctness.

The input data must have one problem column and one gold-answer column. By default, the expected columns are:

```json
{"question": "Jan has 3 apples and buys 4 more. How many apples?", "answer": "#### 7"}
{"question": "A box has 12 pens split equally among 3 students. How many each?", "answer": "#### 4"}
```

CSV files with the same columns are also supported:

```csv
question,answer
"Jan has 3 apples and buys 4 more. How many apples?","#### 7"
```

If your columns have different names, pass `--question-column` and `--answer-column`.

### Local JSONL/CSV Data

```bash
python scripts/plot_custom_math_pca_trajectory.py \
  --model-id google/gemma-4-E2B-it \
  --data-file /path/to/math_data.jsonl \
  --question-column question \
  --answer-column answer \
  --output-dir outputs/math_demo \
  --sample-size 200 \
  --max-per-group 75 \
  --dtype bfloat16
```

### Hugging Face Dataset

```bash
python scripts/plot_custom_math_pca_trajectory.py \
  --model-id google/gemma-4-E2B-it \
  --dataset-name openai/gsm8k \
  --dataset-config main \
  --split train \
  --output-dir outputs/gsm8k_demo \
  --sample-size 200 \
  --max-per-group 75 \
  --dtype bfloat16
```

### Cache Directories

The workflow is environment-agnostic. If a machine needs explicit cache paths, pass them directly:

```bash
python scripts/plot_custom_math_pca_trajectory.py \
  --model-id google/gemma-4-E2B-it \
  --data-file /path/to/math_data.jsonl \
  --output-dir outputs/math_demo \
  --hf-home /scratch/$USER/.cache/huggingface \
  --datasets-cache /scratch/$USER/.cache/huggingface/datasets \
  --transformers-cache /scratch/$USER/.cache/huggingface/transformers \
  --mpl-cache /scratch/$USER/.cache/matplotlib
```

These flags set `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and `MPLCONFIGDIR` for the run. If omitted, the standard library defaults are used.

### Outputs

For `--output-dir outputs/math_demo`, the command writes:

- `outputs/math_demo/inference/custom_math_seed0.jsonl`: prompts, model outputs, parsed numeric answers, gold answers, correctness, and token metadata.
- `outputs/math_demo/activations/custom_math_seed0_all_activations.npz`: final-generated-token residual-stream activations across layers.
- `outputs/math_demo/plots/custom_math_trajectory_pca_correctness.png`: PCA trajectory plot.

The plot treats each example as a residual-stream trajectory through layers. Green lines are correct examples, red lines are incorrect examples, circles mark the first layer state, squares mark the final layer state, and darker segments indicate later layer progression. Layers are normalized before PCA by default; pass `--no-layer-normalize` to disable that.

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
