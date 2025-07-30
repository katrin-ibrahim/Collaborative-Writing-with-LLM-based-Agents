#!/usr/bin/env python3
"""
Single entry point for all baseline experiments.
Supports both Ollama and local model backends.
"""
from baseline_runner_base import run_baseline_experiment
from cli_args import parse_arguments
from runner_factory import create_runner


def main():
    """
    Single main function for all baseline experiments.
    No need for separate local/ollama entry points.
    """
    args = parse_arguments()

    # Create appropriate runner based on backend
    runner_class, runner_name = create_runner(args.backend)

    # Run the experiment using the single canonical function
    return run_baseline_experiment(args, runner_class, runner_name)


if __name__ == "__main__":
    exit(main())
