"""
Experiment Generator for Retrieval Manager Comparison
"""

import argparse
import itertools
import shlex
from datetime import datetime

# =================== Search Space ===================
backends = ["ollama"]
methods = [["rag", "direct", "writer_only", "writer_reviewer", "writer_reviewer_tom", "storm"]]
num_topics = [100]
model_configs = ["small_writer", "balanced_writer", "large_writer"]
retrieval_managers = ["hybrid"]
semantic_filtering = [True]


# =================== Helpers ===================
def make_dir_name(cfg: dict) -> str:
    """Make unique dir name based on args."""
    name_parts = [
        f"n{cfg['num_topics']}",
        f"c-{cfg['model_config']}",
    ]
    if cfg["override_model"]:
        name_parts.append(f"om-{cfg['override_model'].replace(':','-')}")
    if cfg["retrieval_manager"]:
        name_parts.append(f"rm-{cfg['retrieval_manager']}")
    if cfg["semantic_filtering"]:
        name_parts.append("sf")
    return "__".join(name_parts)


def make_command(cfg: dict, exp_name: str) -> str:
    """Make python command string based on args."""
    cmd = ["python", "-m", "src.main"]
    cmd += ["--backend", cfg["backend"]]
    cmd += ["--methods"] + cfg["methods"]
    cmd += ["--num_topics", str(cfg["num_topics"])]
    cmd += ["--model_config", cfg["model_config"]]
    if cfg["override_model"]:
        cmd += ["--override_model", cfg["override_model"]]
    if cfg["retrieval_manager"]:
        cmd += ["--retrieval_manager", cfg["retrieval_manager"]]
    # Always write --semantic_filtering true/false explicitly
    if cfg["semantic_filtering"] is not None:
        val = "true" if cfg["semantic_filtering"] else "false"
        cmd += ["--semantic_filtering", val]
    # Use experiment_name - OutputManager will handle backend directory structure
    cmd += ["--experiment_name", exp_name]
    return " ".join(shlex.quote(arg) for arg in cmd)


# =================== Generate ===================
def main():
    parser = argparse.ArgumentParser(description="Generate experiment scripts")
    parser.add_argument(
        "--exp_name", "-n", default="exp", help="Experiment name prefix (default: exp)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="run_experiments.sh",
        help="Output script file (default: run_experiments.sh)",
    )
    args = parser.parse_args()

    with open(args.output, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated experiment script: {args.exp_name}\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# \n\n")

        exp_counter = 1

        for backend, meths, n, mc, rm, sf in itertools.product(
            backends,
            methods,
            num_topics,
            model_configs,
            retrieval_managers,
            semantic_filtering,
        ):
            # Skip invalid combos
            if "storm" in meths and backend == "slurm":
                continue
            if backend == "ollama" and mc == "slurm":
                continue
            if backend == "slurm" and mc == "ollama_localhost":
                continue

            cfg = {
                "backend": backend,
                "methods": meths,
                "num_topics": n,
                "model_config": mc,
                "override_model": None,
                "retrieval_manager": rm,
                "semantic_filtering": sf,
            }

            dir_name = make_dir_name(cfg)
            full_exp_name = f"{args.exp_name}_{exp_counter:03d}_{dir_name}_{datetime.now().strftime('%d-%m_%H:%M')}"
            cmd = make_command(cfg, full_exp_name)

            f.write(f'echo "Starting experiment {exp_counter}: {full_exp_name}"\n')
            f.write(
                f"echo \"Results will be saved to: results/{cfg['backend']}/{full_exp_name}\"\n"
            )
            f.write(f"{cmd}\n")
            f.write(
                f'echo "Experiment {exp_counter} baseline completed, running evaluation..."\n'
            )
            f.write(
                f"python -m src.evaluation \"results/{cfg['backend']}/{full_exp_name}\"\n"
            )
            f.write(f'echo "Evaluation completed, running analysis..."\n')
            f.write(
                f"python -m src.analysis \"results/{cfg['backend']}/{full_exp_name}\"\n"
            )
            f.write(
                f'echo "Completed experiment {exp_counter} with evaluation and analysis"\n'
            )
            f.write('echo "---"\n\n')

            exp_counter += 1

        f.write(f'echo "All experiments completed! Total: {exp_counter-1}"\n')

    print(f"Generated {exp_counter-1} experiments in {args.output}")
    print(f"Run with: chmod +x {args.output} && ./{args.output}")


if __name__ == "__main__":
    main()
