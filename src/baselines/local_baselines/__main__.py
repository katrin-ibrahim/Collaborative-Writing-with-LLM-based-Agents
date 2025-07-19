#!/usr/bin/env python3
"""
Main runner for local model-based baseline experiments.
Uses locally hosted Qwen models instead of Ollama.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.local_baselines.cli_args import parse_arguments
from src.local_baselines.runner import LocalBaselineRunner

from baselines.baseline_runner_base import run_baseline_experiment


def main():
    args = parse_arguments()
    return run_baseline_experiment(args, LocalBaselineRunner, "Local")


if __name__ == "__main__":
    exit(main())
