#!/usr/bin/env python3
"""
Main runner for Ollama-based baseline experiments.
Clean architecture without HPC workarounds.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.ollama_baselines.cli_args import parse_arguments  # Import from new file
from src.ollama_baselines.runner import BaselineRunner

from baselines.baseline_runner_base import run_baseline_experiment


def main():
    args = parse_arguments()
    return run_baseline_experiment(args, BaselineRunner, "Ollama")


if __name__ == "__main__":
    exit(main())
