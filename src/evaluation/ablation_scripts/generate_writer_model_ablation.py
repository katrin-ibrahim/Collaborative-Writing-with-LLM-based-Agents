#!/usr/bin/env python3
"""

- `-en` exactly matches the results directory name.
- Results directories are NOT nested; they are created as: results/<dir_name>
- Uses --model_config model_phase1_best and overrides writer/reviewer only.
- Each experiment writes logs to results/<dir_name>/run.log
"""

import argparse
from datetime import datetime
from pathlib import Path

# model families (default)
FAMILIES = {
    "gpt_oss": ["gpt-oss:120b", "gpt-oss:20b"],
    "deepseek": ["deepseek-r1:70b", "deepseek-r1:14b"],
    "qwen": ["qwen2.5:72b", "qwen2.5:32b"],
}


def clean_name(model: str) -> str:
    # filesystem-friendly short name
    return model.replace(":", "_").replace("/", "_").replace(".", "-")


def make_exp_name(model: str, exp_prefix: str) -> str:
    return f"{exp_prefix}_{clean_name(model)}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase2 run script (writer model sweep)"
    )
    parser.add_argument(
        "--exp_prefix",
        "-p",
        default="writer_sweep",
        help="Prefix for experiment IDs (optional)",
    )
    parser.add_argument(
        "--backend", "-b", default="ollama", help="Backend to pass to runs"
    )
    parser.add_argument(
        "--num_topics",
        "-n",
        type=int,
        default=7,
        help="Number of topics (passed to main if desired)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="writer",
        help="Experiment mode (writer/reviewer), currently only 'writer' is supported",
    )

    output_dir = f"src/evaluation/ablation_scripts/{parser.parse_args().mode}_sweep_experiments.sh"
    args = parser.parse_args()

    all_models = sum(FAMILIES.values(), [])

    out_path = Path(output_dir).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exp_prefix = f"{args.mode}_sweep_2"

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"# Generated Phase2 writer sweep: {datetime.now().isoformat()}\n")
        f.write(f"# Total experiments: {len(all_models)}\n\n")

        for i, (model) in enumerate(all_models, start=1):
            dir_name = make_exp_name(model, exp_prefix)
            # experiment name equals dir_name (no nesting)
            exp_name = dir_name
            result_dir = f"results/ollama/{dir_name}"

            f.write(f'echo "[{i}/{len(all_models)}] Running experiment: {exp_name}"\n')
            f.write(f'echo " -> model: {model}"\n')
            f.write(f'mkdir -p "{result_dir}"\n')
            model_arg = "writer_model" if args.mode == "writer" else "reviewer_model"

            cmd_parts = [
                "python -m src.main",
                f'-en "{exp_name}"',
                "--model_config model_phase1_best",
                f'--{model_arg} "{model}"',
                f"--backend {args.backend}",
                '--override_model "qwen2.5:32b"',
                "--methods writer_reviewer",
            ]
            # include num_topics only if passed (kept simple)
            if args.num_topics:
                cmd_parts.append(f"--num_topics {args.num_topics}")

            cmd = " ".join(cmd_parts)
            # write generation -> tee run.log, then evaluation append
            # write generation -> log only (no terminal spam)
            f.write(f'{cmd} >> "{result_dir}/main_run.log" 2>&1\n')
            f.write(
                f'echo "Generation finished; running evaluation for {exp_name}..."\n'
            )
            f.write(
                f'python -m src.evaluation "{result_dir}" >> "{result_dir}/eval_run.log" 2>&1\n'
            )
            f.write(
                f'echo "[{i}/{len(all_models)}] Completed: {exp_name}  Results: {result_dir}"\n'
            )

        f.write(f'echo "All {len(all_models)} experiments completed."\n')

    # make executable
    try:
        mode = out_path.stat().st_mode
        out_path.chmod(mode | 0o111)
    except Exception:
        pass

    print(f"Wrote {out_path} (executable). Run with: ./{out_path.name}")


if __name__ == "__main__":
    main()
