# Arabic Dialect GSM8K Residual-Stream Pipeline

This repo is currently set up for one workflow:

- run `Qwen` on `afaji/ArabicGSM8k`
- filter to `dialect=EGY`
- run both subsets:
  - `all replacement`
  - `no replacement`
- save model outputs, residual-stream activations, and PCA comparison plots

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

You need a Hugging Face token with access to `afaji/ArabicGSM8k`.

## Main Run

For a GPU machine, this is the main one-command run:

```bash
PYTHONPATH=src python3 scripts/run_arabic_gsm8k_egy_pair.py \
  --hf-token "$HF_TOKEN" \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --device-map auto \
  --output-dir outputs/arabic_gsm8k_qwen2_5_7b
```

What it does:

- uses dataset `afaji/ArabicGSM8k`
- uses split `test`
- filters `dialect=EGY`
- runs both `all replacement` and `no replacement`
- saves inference JSONL files
- saves final-token residual-stream activations across layers
- writes the PCA plot gallery automatically

## Smaller Smoke Run

If you want a lighter first pass:

```bash
PYTHONPATH=src python3 scripts/run_arabic_gsm8k_egy_pair.py \
  --hf-token "$HF_TOKEN" \
  --model-id Qwen/Qwen3-0.6B \
  --dtype float32 \
  --device-map cpu \
  --sample-size 50 \
  --output-dir outputs/arabic_gsm8k_qwen3_0_6b_smoke
```

## Important Files

- `scripts/run_arabic_gsm8k_egy_pair.py`
  Runs both subset settings in one command.
- `scripts/plot_custom_math_postprocess.py`
  Regenerates the PCA gallery from saved activations without rerunning inference.
- `src/geometric_complexity_scaling/custom_math.py`
  Dataset loading, filtering, inference logging, and activation saving.
- `src/geometric_complexity_scaling/plotting.py`
  PCA plotting and postprocessing gallery logic.

## Outputs

For an output directory like `outputs/arabic_gsm8k_qwen2_5_7b`, the run writes:

- `inference/custom_math_all-replacement_dialect-egy_seed0.jsonl`
- `inference/custom_math_no-replacement_dialect-egy_seed0.jsonl`
- `activations/custom_math_all-replacement_dialect-egy_seed0_all_activations.npz`
- `activations/custom_math_no-replacement_dialect-egy_seed0_all_activations.npz`
- `plots/*.png`

Each JSONL row includes:

- `prompt`
- `raw_output`
- `parsed_answer`
- `target`
- `correct`
- `generated_len`
- `max_new_tokens`
- `hit_max_new_tokens`
- `source_row`

Each activation file includes:

- `activations`
- `row_ids`
- `correct`
- `outcome_group`
- `generated_lengths`

## Plot Gallery

When both subsets are present, the repo writes these comparison plots:

- per-subset local PCA plot for `all replacement`
- per-subset local PCA plot for `no replacement`
- 4-line shared PCA comparison:
  - `no replacement correct`
  - `no replacement incorrect`
  - `all replacement correct`
  - `all replacement incorrect`
- replacement-only comparison
- correctness-only comparison
- agreement-only comparison
  Rows where both settings produce the same final parsed answer.
- disagreement-only comparison
  Rows where the two settings produce different final parsed answers.

Agreement/disagreement uses the math scorer’s numeric equivalence on `parsed_answer`, not exact raw-text matching.

## Rebuild Plots Without Rerunning Inference

```bash
PYTHONPATH=src python3 scripts/plot_custom_math_postprocess.py \
  --all-replacement-activation outputs/arabic_gsm8k_qwen2_5_7b/activations/custom_math_all-replacement_dialect-egy_seed0_all_activations.npz \
  --no-replacement-activation outputs/arabic_gsm8k_qwen2_5_7b/activations/custom_math_no-replacement_dialect-egy_seed0_all_activations.npz \
  --output-dir outputs/arabic_gsm8k_qwen2_5_7b \
  --plot-prefix custom_math_seed0_dialect-egy \
  --max-per-group 75
```

Use `--max-per-group 0` to keep every saved example in each label group.

## Notes

- The current default longer generation budget is `512` new tokens.
- The dataset config names use spaces:
  - `all replacement`
  - `no replacement`
- The current workflow assumes the dataset columns:
  - `ID`
  - `question`
  - `answer`
  - `dialect`
