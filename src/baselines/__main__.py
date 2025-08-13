#!/usr/bin/env python3
"""
Single entry point for all baseline experiments.
Supports both Ollama and local model backends.
"""
import os

from src.baselines.baseline_runner_base import run_baseline_experiment
from src.baselines.cli_args import parse_arguments
from src.baselines.runner_factory import create_runner


def setup_hf_cache_for_backend(backend: str):
    """Set HuggingFace cache directory based on backend mode."""
    if backend == "local":
        # Local backend might have restricted permissions
        cache_dir = "/tmp/hf_cache_local"
    else:
        # Ollama backend, use standard cache
        cache_dir = os.path.expanduser("~/.cache/huggingface")

    # Test write permissions and fallback if needed
    try:
        os.makedirs(cache_dir, exist_ok=True)
        test_file = os.path.join(cache_dir, "test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (OSError, PermissionError):
        cache_dir = f"/tmp/hf_cache_{os.getenv('USER', 'unknown')}"
        os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables that all HF libraries will use
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir


def main():
    """
    Single main function for all baseline experiments.
    No need for separate local/ollama entry points.
    """
    args = parse_arguments()

    setup_hf_cache_for_backend(args.backend)

    # Create appropriate runner based on backend
    runner_class, runner_name = create_runner(args.backend)

    # Run the experiment using the single canonical function
    return run_baseline_experiment(args, runner_class, runner_name)


if __name__ == "__main__":
    exit(main())
