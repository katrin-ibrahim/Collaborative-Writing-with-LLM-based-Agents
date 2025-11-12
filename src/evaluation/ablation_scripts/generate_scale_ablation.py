#!/usr/bin/env python3
"""
Generate a shell script that runs the model-size ablation experiments.

Saves: run_model_ablation.sh (by default) and makes it executable.

Usage:
    python generate_model_ablation_sh.py
    python generate_model_ablation_sh.py  --num_topics 10
"""

import argparse
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------
# Experiment config (copied from your original)
# ---------------------------
@dataclass
class ExperimentConfig:
    name: str
    task_model: str
    model_size: str
    description: str


def get_task_models() -> List[str]:
    return [
        "query_generation_model",
        "create_outline_model",
        "section_selection_model",
        "revision_model",
        "revision_batch_model",
        "self_refine_model",
    ]


def generate_experiments(include_baseline: bool = True) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = []

    if include_baseline:
        experiments.append(
            ExperimentConfig(
                name="baseline_32b",
                task_model="all",
                model_size="32b",
                description="Baseline: all tasks use 32b",
            )
        )

    for task_model in get_task_models():
        task_name = task_model.replace("_model", "")
        experiments.append(
            ExperimentConfig(
                name=f"{task_name}_14b",
                task_model=task_model,
                model_size="14b",
                description=f"Downgrade {task_name} to 14b (others at 32b)",
            )
        )
        experiments.append(
            ExperimentConfig(
                name=f"{task_name}_7b",
                task_model=task_model,
                model_size="7b",
                description=f"Downgrade {task_name} to 7b (others at 32b)",
            )
        )
    return experiments


# ---------------------------
# Mapping for task flags (same as original)
# ---------------------------
CLI_ARG_MAP = {
    "query_generation_model": "-qgm",
    "create_outline_model": "-oum",
    "section_selection_model": "-ssm",
    "writer_model": "-wtm",
    "revision_model": "-rvm",
    "revision_batch_model": "-rbm",
    "self_refine_model": "-srm",
    "reviewer_model": "-rwm",
}


# ---------------------------
# Build the shell command string for a given experiment
# ---------------------------
def build_command(
    exp: ExperimentConfig, num_topics: int, output_base: Path
) -> Tuple[str, str]:
    experiment_name = f"model_ablation_{exp.name}"
    outdir = output_base / experiment_name

    parts = [
        "python -m src.main",
        "-m",
        "writer_reviewer",
        "-n",
        str(num_topics),
        "-b",
        "ollama",
        "-om",
        "qwen2.5:32b",
        "--experiment_name",
        experiment_name,
    ]

    if exp.task_model != "all":
        cli_flag = CLI_ARG_MAP.get(exp.task_model)
        if cli_flag:
            parts += [cli_flag, f"qwen2.5:{exp.model_size}"]
        else:
            parts += [f"--{exp.task_model}", f"qwen2.5:{exp.model_size}"]

    cmd = " ".join(parts)
    cmd = f'mkdir -p "{outdir}" && {cmd} 2>&1 | tee -a "{outdir}/run.log"'
    return cmd, str(outdir)


# ---------------------------
# Write shell script
# ---------------------------
def write_shell(
    experiments: List[ExperimentConfig],
    num_topics: int,
    output_path: Path,
    output_base: Path,
):
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Experiments: {len(experiments)}",
        "",
    ]
    lines = header[:]
    idx = 1
    for exp in experiments:
        cmd, outdir = build_command(exp, num_topics, output_base)
        lines.append(f'echo "Starting experiment {idx}: {exp.name}"')
        lines.append(f'echo "Description: {exp.description}"')
        lines.append(f'echo "Results will be saved to: {outdir}"')
        lines.append(cmd)
        lines.append(
            f'echo "Experiment {idx} generation completed, running evaluation..."'
        )
        lines.append(
            f'python -m src.evaluation "{outdir}" 2>&1 | tee -a "{outdir}/run.log"'
        )
        lines.append(f'echo "Completed experiment {idx} with evaluation"')
        lines.append('echo "---"')
        lines.append("")  # blank line for readability
        idx += 1

    lines.append(f'echo "All experiments completed! Total: {idx-1}"')
    # write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # make executable
    try:
        mode = output_path.stat().st_mode
        output_path.chmod(mode | 0o111)
    except Exception as e:
        print(f"Warning: could not make {output_path} executable: {e}")


# ---------------------------
# CLI: simple & explicit
# ---------------------------
def main():
    parser = argparse.ArgumentParser(prog="generate_model_ablation_sh")
    parser.add_argument(
        "--num_topics",
        "-n",
        type=int,
        default=10,
        help="Number of topics per experiment (default: 10)",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("results/ollama"),
        help="Base results directory (default: results/ollama)",
    )
    parser.add_argument(
        "--no_baseline", action="store_true", help="Do not include baseline experiment"
    )

    args = parser.parse_args()

    output_dir = "src/evaluation/ablation_scripts/run_model_ablation.sh"

    experiments = generate_experiments(include_baseline=not args.no_baseline)
    print(f"Generated {len(experiments)} experiment entries (will write shell script)")

    write_shell(
        experiments,
        num_topics=args.num_topics,
        output_path=Path(output_dir),
        output_base=args.output_base,
    )

    print(f"Wrote shell script: {args.output.resolve()}")
    print(f"Make executable & run: chmod +x {args.output.name} && ./{args.output.name}")


if __name__ == "__main__":
    main()
