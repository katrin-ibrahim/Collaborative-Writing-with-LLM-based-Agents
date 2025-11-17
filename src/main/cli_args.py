"""
CLI argument parsing for collaborative writing main.
Based on baselines CLI but focused on collaborative methods.
"""

import argparse
from pathlib import Path


def _get_available_model_configs() -> list[str]:
    """
    Dynamically discover available model configuration files.

    Returns:
        List of config names (without 'model_' prefix and '.yaml' extension)
    """
    config_dir = Path(__file__).parent.parent / "config" / "configs"
    if not config_dir.exists():
        return ["general"]

    config_files = list(config_dir.glob("model_*.yaml"))
    config_names = [f.stem.replace("model_", "") for f in config_files]
    return sorted(config_names) if config_names else ["general"]


def _get_available_collaboration_configs() -> list[str]:
    """
    Dynamically discover available collaboration configuration files.

    Returns:
        List of config names (without 'collaboration_' prefix and '.yaml' extension)
    """
    config_dir = Path(__file__).parent.parent / "config" / "configs"
    if not config_dir.exists():
        return ["default"]

    config_files = list(config_dir.glob("collaboration_*.yaml"))
    config_names = [f.stem.replace("collaboration_", "") for f in config_files]
    return sorted(config_names) if config_names else ["default"]


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
        choices=[
            "writer",
            "writer_reviewer",
            "writer_reviewer_tom",
            "direct",
            "rag",
            "storm",
        ],
        default=["writer_only"],
        help="Methods to run (default: writer_only). Note: STORM only works with --backend ollama.",
    )

    # =================== Common Configuration ===================
    config_group = parser.add_argument_group("Configuration Options")

    available_configs = _get_available_model_configs()

    config_group.add_argument(
        "--model_config",
        "-c",
        default="general",
        help=f"Model configuration preset (default: general). Available at startup: {', '.join(available_configs)}. Dynamically generated configs are also supported.",
    )

    config_group.add_argument(
        "--override_model",
        "-om",
        help="Override model to use for all tasks instead of task-specific models (e.g., qwen2.5:7b, qwen2.5:14b, qwen2.5:32b, gpt-oss:20b)",
    )

    # =================== Granular Model Parameters ===================
    model_group = parser.add_argument_group("Granular Model Configuration")

    model_group.add_argument(
        "--outline_model",
        "-oum",
        help="Model for creating article outlines",
    )

    model_group.add_argument(
        "--research_model",
        "-resm",
        help="Model for selecting relevant chunks, querying, and research synthesis",
    )

    model_group.add_argument(
        "--writer_model",
        "-wtm",
        help="Model for writing content",
    )

    model_group.add_argument(
        "--revision_model",
        "-rvm",
        help="Model for revising sections based on feedback",
    )
    model_group.add_argument(
        "--self_refine_model",
        "-srm",
        help="Model for self-refinement",
    )

    model_group.add_argument(
        "--reviewer_model",
        "-rwm",
        help="Model for holistic review and feedback generation",
    )

    # =================== Granular Retrieval Parameters ===================
    retrieval_group = parser.add_argument_group("Retrieval Configuration Override")

    retrieval_group.add_argument(
        "--retrieval_manager",
        "-rm",
        choices=[
            "wiki",
            "faiss",
        ],
        default="wiki",
        help="Retrieval manager type (overrides config file)",
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
        help="Experiment name for auto-generated output directory (e.g., 'gpt_test')",
    )

    # Collaboration-specific options
    available_collab_configs = _get_available_collaboration_configs()

    parser.add_argument(
        "--collaboration_config",
        "-cc",
        type=str,
        default="default",
        help=f"Collaboration configuration preset to use (default: default). Available at startup: {', '.join(available_collab_configs)}. Dynamically generated configs are also supported.",
    )

    parser.add_argument(
        "--max_iterations",
        "-i",
        type=int,
        help="Override max collaboration iterations",
    )

    parser.add_argument(
        "--min_iterations",
        type=int,
        help="Override min collaboration iterations (minimum before convergence check)",
    )

    parser.add_argument(
        "--convergence_threshold",
        "-ct",
        type=float,
        help="Override convergence threshold (resolution_rate_threshold)",
    )

    parser.add_argument(
        "--resolution_rate_threshold",
        type=float,
        help="Override resolution rate threshold for convergence",
    )

    parser.add_argument(
        "--stall_tolerance",
        type=int,
        help="Override stall tolerance (consecutive low-improvement iterations allowed)",
    )

    parser.add_argument(
        "--min_improvement",
        type=float,
        help="Override minimum improvement per iteration (e.g., 0.02 for 2%%)",
    )

    parser.add_argument(
        "--small_tail_max",
        type=int,
        help="Override small tail max (max remaining low/medium items for convergence)",
    )
    parser.add_argument(
        "--writing_mode",
        "-wm",
        choices=["section", "article"],
        default="section",
        help="Set the writing mode (default: section)",
    )
    parser.add_argument(
        "--revise_mode",
        "-revm",
        choices=["section", "pending"],
        default="pending",
        help="Set the revise mode: 'pending' = batch mode (fast, 1 LLM call), 'section' = sequential (slow, 1 call per section)",
    )
    parser.add_argument(
        "--no_self_refine",
        "-nsr",
        action="store_false",
        dest="no_self_refine",
        help="Disable self-refinement by writers (default: self-refine enabled)",
    )

    parser.add_argument(
        "--no_reviewer_grounding",
        "-nrg",
        action="store_true",
        dest="no_reviewer_grounding",
        help="Disable grounding the reviewer with research (default: reviewer grounding enabled)",
    )

    # Parse arguments
    args = parser.parse_args()
    # =================== Post-Processing & Validation ===================

    # Auto-set model config based on backend if using defaults
    if (
        args.model_config != "slurm"
        and args.model_config != "slurm_thinking"
        and args.backend == "slurm"
    ):
        args.model_config = "slurm"
        print(
            f"Auto-setting model config to '{args.model_config}' for backend '{args.backend}'"
        )

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
