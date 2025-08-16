"""
Single unified CLI argument parser for all baseline experiments.
Supports both Ollama and local backends through --backend flag.
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for baseline experiments.
    Single parser handles both local and Ollama backends.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with local or Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods with Ollama (default backend)
  %(prog)s --methods direct storm rag --num_topics 10

  # Run only direct + rag with SLURM models
  %(prog)s --backend slurm --methods direct rag --num_topics 5

  # Resume experiment from checkpoint
  %(prog)s --resume_dir /path/to/experiment/dir

  # Use single model for all tasks
  %(prog)s --override_model qwen2.5:14b --methods direct rag --num_topics 5

  # Use GPT-OSS-20B for all tasks
  %(prog)s --backend ollama --override_model gpt-oss:20b --methods direct --num_topics 3
        """,
    )

    # =================== Core Arguments ===================
    parser.add_argument(
        "--backend",
        "-b",
        choices=["ollama", "slurm"],
        default="ollama",
        help="Model execution backend: 'ollama' for API-based (localhost/remote), 'slurm' for direct model execution",
    )

    parser.add_argument(
        "--num_topics",
        "-n",
        type=int,
        default=5,
        help="Number of topics to evaluate (default: 5)",
    )

    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=["direct"],
        choices=["direct", "storm", "rag", "agentic", "collaborative"],
        help="Methods to run (default: direct). Note: STORM only works with --backend ollama. 'collaborative' uses writer-reviewer collaboration.",
    )

    # =================== Common Configuration ===================
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument(
        "--model_config",
        "-c",
        default="ollama_localhost",
        choices=["ollama_localhost", "ollama_ukp", "slurm", "slurm_thinking"],
        help="Model configuration preset (default: ollama_localhost)",
    )

    config_group.add_argument(
        "--override_model",
        "-om",
        help="Override model to use for all tasks instead of task-specific models (e.g., qwen2.5:7b, qwen2.5:14b, qwen2.5:32b, gpt-oss:20b)",
    )

    # =================== Granular Retrieval Parameters ===================
    retrieval_group = parser.add_argument_group("Retrieval Configuration Override")

    retrieval_group.add_argument(
        "--retrieval_manager",
        "-rm",
        choices=["wiki", "bm25_wiki", "faiss_wiki"],
        help="Retrieval manager type (overrides config file)",
    )

    retrieval_group.add_argument(
        "--semantic_filtering",
        "-sf",
        action="store_true",
        help="Enable semantic filtering for retrieval results",
    )

    retrieval_group.add_argument(
        "--use_wikidata_enhancement",
        "-uw",
        action="store_true",
        help="Enable Wikidata retrieval enhancement",
    )

    # =================== Output & Debugging ===================
    output_group = parser.add_argument_group("Output & Debugging Options")
    output_group.add_argument(
        "--log_level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    output_group.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode (saves intermediate files)",
    )

    output_group.add_argument(
        "--resume_dir", "-r", type=str, help="Resume from specific experiment directory"
    )

    output_group.add_argument(
        "--output_dir", "-o", type=str, help="Custom output directory for results"
    )

    output_group.add_argument(
        "--experiment_name",
        "-en",
        type=str,
        help="Experiment name for auto-generated output directory (e.g., 'semantic_filtering_test')",
    )

    output_group.add_argument(
        "--auto_name",
        "-an",
        action="store_true",
        help="Auto-generate output directory name based on experiment parameters",
    )

    # Parse arguments
    args = parser.parse_args()

    # =================== Post-Processing & Validation ===================

    # Convert resume_dir to resume flag
    if args.resume_dir:
        args.resume = True

    # Validate method combinations
    if "storm" in args.methods and args.backend == "slurm":
        parser.error(
            "STORM method is not supported with --backend slurm. Use --backend ollama for STORM."
        )

    # Handle list retrieval configs request (remove this feature since we removed --retrieval_config)
    # This was used for listing available YAML configs, but we're now using defaults + CLI overrides

    return args


def validate_args(args) -> bool:
    """
    Additional validation after parsing.

    Args:
        args: Parsed arguments

    Returns:
        True if valid, False otherwise
    """
    # Check if resume directory exists
    if hasattr(args, "resume_dir") and args.resume_dir:
        from pathlib import Path

        resume_path = Path(args.resume_dir)
        if not resume_path.exists():
            print(f"Error: Resume directory does not exist: {resume_path}")
            return False
        if not (resume_path / "results.json").exists():
            print(f"Error: No results.json found in resume directory: {resume_path}")
            return False

    # Validate method-backend combinations
    if args.backend == "slurm" and "storm" in args.methods:
        print("Error: STORM is not supported with slurm backend")
        return False

    return True


if __name__ == "__main__":
    # Test the argument parser
    args = parse_arguments()
    if validate_args(args):
        print("✅ Arguments parsed successfully:")
        print(f"  Backend: {args.backend}")
        print(f"  Methods: {args.methods}")
        print(f"  Topics: {args.num_topics}")
    else:
        print("❌ Argument validation failed")
        exit(1)
