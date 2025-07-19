# FILE: cli_args.py
"""
Unified CLI argument parser for baseline experiments.
Works with both ollama and local model backends.
"""
import argparse

from typing import Any, Dict


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments for baseline experiments with flexible backend support.

    Returns:
        Dict of parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with local or Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on 10 topics with Ollama
  %(prog)s --backend ollama --num_topics 10

  # Run only STORM on 5 topics with local models
  %(prog)s --backend local --methods storm --num_topics 5

  # Use custom Ollama host
  %(prog)s --backend ollama --ollama_host http://localhost:11434/ --num_topics 5

  # Use custom local model path
  %(prog)s --backend local --model_path /path/to/models --num_topics 5
        """,
    )

    # Core arguments
    parser.add_argument(
        "--backend",
        choices=["ollama", "local"],
        default="local",
        help="Model backend to use: ollama or local",
    )
    parser.add_argument(
        "-n", "--num_topics", type=int, default=5, help="Number of topics to evaluate"
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=["direct"],
        choices=["direct", "storm", "rag"],
        help="Methods to run",
    )
    parser.add_argument(
        "--dataset_path",
        default="dataset_report.json",
        help="Path to dataset with topics",
    )

    # Backend-specific arguments
    parser.add_argument(
        "-H",
        "--ollama_host",
        default="http://10.167.31.201:11434/",
        help="Ollama server URL (only used with --backend ollama)",
    )
    parser.add_argument(
        "-p",
        "--model_path",
        default="models/",
        help="Path to local models directory (only used with --backend local)",
    )
    parser.add_argument(
        "-s",
        "--model_size",
        choices=["7b", "14b", "32b", "72b"],
        default="32b",
        help="Which model size to use (only used with --backend local)",
    )

    # Common arguments
    parser.add_argument(
        "-c",
        "--model_config",
        default="config/models.yaml",
        help="Model configuration file",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode (saves intermediate files)",
    )
    parser.add_argument(
        "-r",
        "--resume_dir",
        type=str,
        help="Resume from specific run directory path",
    )

    args = parser.parse_args()

    # Convert args to resume if resume_dir is specified
    if args.resume_dir:
        args.resume = True

    # Process model size to update default model in config (for local backend)
    if args.backend == "local":
        args.model_name = f"qwen2.5:{args.model_size}"

    return args
