#!/usr/bin/env python3
"""
Main entry point for collaborative writing experiments.
Supports both Ollama and SLURM backends with factory pattern.
"""


import logging
import os

from src.runners.cli_args import parse_arguments
from src.runners.factory import create_runner
from src.utils.data import FreshWikiLoader
from src.utils.experiment import setup_output_directory
from src.utils.io import OutputManager

logger = logging.getLogger(__name__)


def setup_hf_cache_for_backend(backend: str):
    """Set HuggingFace cache directory based on backend mode."""
    if backend == "slurm":
        cache_dir = "/tmp/hf_cache_slurm"
    else:
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
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir


def main():
    """Main entry point for collaborative writing experiments."""

    args = parse_arguments()

    # Setup HuggingFace cache
    setup_hf_cache_for_backend(args.backend)

    # Create runner instance
    model_config, retrieval_config, collaboration_config = load_configurations(args)

    # Setup output directory
    output_dir = setup_output_directory(args)
    output_manager = OutputManager(str(output_dir), debug_mode=args.debug)

    # ONE LINE - Factory handles everything
    runner = create_runner(
        backend=args.backend,
        model_config=model_config,
        output_manager=output_manager,
        retrieval_config=retrieval_config,
        collaboration_config=collaboration_config,
    )

    # Load topics
    freshwiki = FreshWikiLoader()
    topics = [entry.topic for entry in freshwiki.load_topics(args.num_topics)]

    # Run experiment
    runner.run(topics, args.methods)


if __name__ == "__main__":
    exit(main())
