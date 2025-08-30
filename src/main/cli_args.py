"""
CLI argument parsing for collaborative writing main.
Based on baselines CLI but focused on collaborative methods.
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for collaborative writing experiments."""

    parser = argparse.ArgumentParser(
        description="Collaborative Writing with LLM-based Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        choices=["writer_only", "writer_reviewer"],
        help="Methods to run (default: direct). Note: STORM only works with --backend ollama.",
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
        choices=[
            "wiki",
            "supabase_faiss",
        ],
        help="Retrieval manager type (overrides config file)",
    )

    retrieval_group.add_argument(
        "--semantic_filtering",
        "-sf",
        choices=["true", "false"],
        default="true",
        help="Enable or disable semantic filtering for retrieval results (default: true)",
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

    parser.add_argument(
        "--dry_run",
        "-dry",
        action="store_true",
        help="Print configuration and exit without running",
    )
    # Advanced options
    parser.add_argument(
        "--max_workers",
        "-mw",
        type=int,
        help="Maximum number of parallel workers (default: auto)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="Random seed for reproducibility",
    )

    # Collaboration-specific options (for future use)

    parser.add_argument(
        "--collaboration_config",
        "-cc",
        type=str,
        default="default",
        choices=["default", "aggressive", "conservative"],
        help="Collaboration configuration preset to use",
    )

    parser.add_argument(
        "--max_iterations",
        "-i",
        type=int,
        help="Override max collaboration iterations",
    )

    parser.add_argument(
        "--convergence_threshold",
        "-ct",
        type=float,
        help="Override convergence threshold",
    )

    # Parse arguments
    args = parser.parse_args()
    # =================== Post-Processing & Validation ===================

    # Convert resume_dir to resume flag
    if args.resume_dir:
        args.resume = True

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

    return True


if __name__ == "__main__":
    # For testing CLI parsing
    args = parse_arguments()
    if validate_args(args):
        print("✅ Arguments parsed successfully:")
        print(f"  Backend: {args.backend}")
        print(f"  Methods: {args.methods}")
        print(f"  Topics: {args.num_topics}")
    else:
        print("❌ Argument validation failed")
        exit(1)
