from __future__ import annotations

import argparse

from geometric_complexity_scaling.custom_math import run_custom_math_pca_trajectory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the paired ArabicGSM8k EGY experiment for both replacement settings and write the PCA gallery."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-name", default="afaji/ArabicGSM8k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", default="EGY")
    parser.add_argument("--output-dir", default="outputs/arabic_gsm8k_qwen2_5_7b")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-per-group", type=int, default=75)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--geometry-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--datasets-cache", default=None)
    parser.add_argument("--transformers-cache", default=None)
    parser.add_argument("--mpl-cache", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    common_kwargs = dict(
        output_dir=args.output_dir,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        split=args.split,
        question_column="question",
        answer_column="answer",
        row_id_column="ID",
        row_filters=[("dialect", args.dialect)],
        sample_size=args.sample_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
        geometry_dtype=args.geometry_dtype,
        max_per_group=args.max_per_group,
        hf_home=args.hf_home,
        hf_token=args.hf_token,
        datasets_cache=args.datasets_cache,
        transformers_cache=args.transformers_cache,
        mpl_cache=args.mpl_cache,
        overwrite=args.overwrite,
    )
    for dataset_config in ("all replacement", "no replacement"):
        run_custom_math_pca_trajectory(dataset_config=dataset_config, **common_kwargs)


if __name__ == "__main__":
    main()
