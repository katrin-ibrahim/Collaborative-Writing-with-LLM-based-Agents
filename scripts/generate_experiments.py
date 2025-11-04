import argparse
import itertools
import shlex
from datetime import datetime

# =================== Search Space ===================
# Note: These values are now used for agentic experiments (src.main)
backends = ["ollama"]
methods_to_test = ["writer_reviewer"]
num_topics = [20]
model_configs = ["balanced_writer"]
writing_modes = ["full_article"]
revise_modes = ["pending_sections", "single_section"]
retrieval_managers = ["wiki"]


# =================== Helpers ===================
def make_dir_name(cfg: dict) -> str:
    """Make unique dir name based on args, adhering to the new verbose schema."""
    # model_config__writing_mode__revise_mode__rm__method__sf
    rev_mode_tag = cfg["revise_mode"].replace(
        "_", "-"
    )  # pending-sections or single-section

    name_parts = [
        cfg["model_config"],
        cfg["method"],
        f"wm-{cfg['writing_mode']}",
        f"revm-{rev_mode_tag}",
        f"rm-{cfg['retrieval_manager']}",
    ]

    # Custom naming convention based on user request (using '--' as separator for clarity)
    return "--".join(name_parts)


def make_command(cfg: dict, exp_name: str) -> str:
    """Make python command string based on args, targeting src.main for agentic methods."""
    cmd = ["python", "-m", "src.main", "-en", exp_name]

    cmd += ["--backend", cfg["backend"]]
    cmd += ["--methods", cfg["method"]]
    cmd += ["--num_topics", str(cfg["num_topics"])]
    cmd += ["--model_config", cfg["model_config"]]

    if cfg["retrieval_manager"]:
        cmd += ["--retrieval_manager", cfg["retrieval_manager"]]

    # Add mode arguments
    cmd += ["--writing_mode", cfg["writing_mode"]]

    # Only include revise_mode argument for iterative methods
    if "reviewer" in cfg["method"]:
        # The revise_mode tag can be used to set the actual argument
        cmd += ["--revise_mode", cfg["revise_mode"]]

    return shlex.join(cmd)


# =================== Generate ===================
def main():
    parser = argparse.ArgumentParser(description="Generate experiment scripts")
    parser.add_argument(
        "--exp_name",
        "-n",
        default="agent_exp",
        help="Experiment name prefix (default: agent_exp)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="run_agent_experiments.sh",
        help="Output script file (default: run_agent_experiments.sh)",
    )
    args = parser.parse_args()

    with open(args.output, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated experiment script for agentic methods: {args.exp_name}\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# \n\n")

        exp_counter = 1

        # Use itertools.product for Cartesian product of all configuration lists
        for backend, method, n, mc, wm, revm, rm in itertools.product(
            backends,
            methods_to_test,
            num_topics,
            model_configs,
            writing_modes,
            revise_modes,
            retrieval_managers,
        ):
            # 1. Skip non-sense combinations (e.g., SLURM + Ollama configs)
            if backend == "ollama" and mc in ["slurm", "slurm_thinking"]:
                continue
            if backend == "slurm" and mc not in ["slurm", "slurm_thinking"]:
                continue

            # 2. Skip irrelevant combinations: revise_mode is irrelevant for writer_only (only 1 iteration)
            # This skips combinations where revise_mode is not the default 'pending_sections'.
            if method == "writer_only" and revm != "pending_sections":
                continue

            cfg = {
                "backend": backend,
                "method": method,
                "num_topics": n,
                "model_config": mc,
                "writing_mode": wm,
                "revise_mode": revm,
                "retrieval_manager": rm,
            }

            dir_name = make_dir_name(cfg)
            full_exp_name = f"{args.exp_name}_{exp_counter:03d}_{dir_name}_{datetime.now().strftime('%d-%m_%H%M')}"
            cmd = make_command(cfg, full_exp_name)

            f.write(f'echo "Starting experiment {exp_counter}: {full_exp_name}"\n')
            f.write(
                f"echo \"Results will be saved to: results/{cfg['backend']}/{full_exp_name}\"\n"
            )
            f.write(f"{cmd}\n")

            # Follow the pipeline steps (generation, evaluation)
            f.write(
                f'echo "Experiment {exp_counter} generation completed, running evaluation..."\n'
            )
            f.write(
                f"python -m src.evaluation \"results/{cfg['backend']}/{full_exp_name}\"\n"
            )
            f.write(f'echo "Completed experiment {exp_counter} with evaluation"\n')
            f.write('echo "---"\n\n')

            exp_counter += 1

        f.write(f'echo "All experiments completed! Total: {exp_counter-1}"\n')

    print(f"Generated {exp_counter-1} experiments in {args.output}")
    print(f"Run with: chmod +x {args.output} && ./{args.output}")


if __name__ == "__main__":
    main()
