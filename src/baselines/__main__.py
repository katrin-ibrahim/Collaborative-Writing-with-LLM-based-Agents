#!/usr/bin/env python3
"""
Unified entry point for baseline experiments.
Supports both Ollama and local model backends.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from baselines.main_runner_base import run_baseline_experiment
from src.baselines.cli_args import parse_arguments
from src.baselines.runner_factory import create_runner


def main():
    """
    Main entry point for baseline experiments.
    Parses arguments, creates appropriate runner, and runs the experiment.

    Returns:
        0 for success, 1 for failure
    """
    args = parse_arguments()

    # Create appropriate runner based on backend
    runner_class, runner_name = create_runner(args.backend)

    # Run the experiment
    return run_baseline_experiment(args, runner_class, runner_name)


if __name__ == "__main__":
    exit(main())
