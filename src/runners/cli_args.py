"""
CLI argument parsing for collaborative writing runners.
Based on baselines CLI but focused on collaborative methods.
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for collaborative writing experiments."""

    parser = argparse.ArgumentParser(
        description="Collaborative Writing with LLM-based Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core experiment arguments
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        choices=["writer_only", "writer_reviewer"],
        default=["writer_only"],
        help="Methods to run",
    )

    parser.add_argument(
        "--num-topics",
        "-n",
        type=int,
        default=5,
        help="Number of topics to process",
    )

    parser.add_argument(
        "--backend",
        "-b",
        choices=["ollama", "slurm"],
        default="ollama",
        help="Backend to use for model inference",
    )

    # Configuration files
    parser.add_argument(
        "--model-config",
        "-mc",
        type=str,
        choices=["ollama_localhost", "ollama_ukp", "slurm", "slurm_thinking"],
        default="ollama_localhost",
        help="Model configuration preset to use",
    )

    parser.add_argument(
        "--collaboration-config",
        "-cc",
        type=str,
        default="default",
        choices=["default", "aggressive", "conservative"],
        help="Collaboration configuration preset to use",
    )

    parser.add_argument(
        "--retrieval-manager",
        "-rm",
        type=str,
        choices=["wiki", "supabase_faiss"],
        help="Retrieval manager to use for knowledge access",
    )

    # Model override
    parser.add_argument(
        "--override-model",
        "-om",
        type=str,
        help="Override model to use for all tasks",
    )

    # Output and logging
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for results (default: auto-generated)",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode with detailed output",
    )

    # Retrieval options
    parser.add_argument(
        "--semantic-filtering",
        "-sf",
        action="store_true",
        help="Enable semantic filtering for retrieved passages",
    )

    # Experiment management
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume experiment from previous checkpoint",
    )
    # Experiment management
    parser.add_argument(
        "--resume_dir",
        "-rd",
        type=str,
        help="Directory to resume experiment from",
    )

    parser.add_argument(
        "--dry-run",
        "-dry",
        action="store_true",
        help="Print configuration and exit without running",
    )

    # Advanced options
    parser.add_argument(
        "--max-workers",
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
        "--max-iterations",
        "-i",
        type=int,
        help="Override max collaboration iterations",
    )

    parser.add_argument(
        "--convergence-threshold",
        "-ct",
        type=float,
        help="Override convergence threshold",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate argument combinations and constraints."""

    # Validate backend-specific requirements
    if args.backend == "slurm" and not args.model_config:
        raise ValueError("SLURM backend requires --model-config to be specified")

    # Validate method-specific requirements
    if (
        "writer_reviewer" in args.methods
        and args.collaboration_config == "conservative"
    ):
        print(
            "Warning: Using conservative config with writer_reviewer may result in minimal collaboration"
        )

    # Validate override options
    if args.max_iterations and args.max_iterations < 1:
        raise ValueError("--max-iterations must be >= 1")

    if args.convergence_threshold and not (0.0 <= args.convergence_threshold <= 1.0):
        raise ValueError("--convergence-threshold must be between 0.0 and 1.0")


def print_configuration(args: argparse.Namespace) -> None:
    """Print experiment configuration for verification."""
    print("\n🔧 Experiment Configuration:")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Backend: {args.backend}")
    print(f"  Topics: {args.num_topics}")
    print(f"  Model config: {args.model_config or 'default'}")
    print(f"  Collaboration config: {args.collaboration_config}")

    if args.retrieval_manager:
        print(f"  Retrieval manager: {args.retrieval_manager}")

    if args.override_model:
        print(f"  Model override: {args.override_model}")

    if args.output_dir:
        print(f"  Output directory: {args.output_dir}")

    print(f"  Debug mode: {args.debug}")
    print(f"  Log level: {args.log_level}")
    print()


if __name__ == "__main__":
    # For testing CLI parsing
    args = parse_arguments()
    validate_arguments(args)
    print_configuration(args)
