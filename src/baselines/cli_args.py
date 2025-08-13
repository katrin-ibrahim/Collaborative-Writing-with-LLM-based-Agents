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

  # Run only direct + rag with local models
  %(prog)s --backend local --methods direct rag --num_topics 5

  # Use custom Ollama host
  %(prog)s --backend ollama --ollama_host http://localhost:11434/ --num_topics 5

  # Use custom local model path and GPU
  %(prog)s --backend local --model_path /path/to/models --device cuda --num_topics 5

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
        choices=["ollama", "local"],
        default="ollama",
        help="Model backend to use (default: ollama)",
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

    # =================== Ollama-Specific Arguments ===================
    ollama_group = parser.add_argument_group("Ollama Backend Options")
    ollama_group.add_argument(
        "--ollama_host",
        "-oh",
        default="http://10.167.31.201:11434/",
        help="Ollama server host URL (default: http://10.167.31.201:11434/)",
    )

    # =================== Local-Specific Arguments ===================
    local_group = parser.add_argument_group("Local Backend Options")
    local_group.add_argument(
        "--device",
        default="auto",
        help="Device for local models: auto, cuda, cpu, mps (default: auto)",
    )

    local_group.add_argument(
        "--model_path", help="Path to local model directory (overrides config)"
    )

    local_group.add_argument(
        "--model_size",
        choices=["7b", "14b", "32b", "72b"],
        default="32b",
        help="Model size for local backend (default: 32b)",
    )

    # =================== Common Configuration ===================
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument(
        "--model_config",
        "-c",
        default="config/models.yaml",
        help="Model configuration file (default: config/models.yaml)",
    )

    config_group.add_argument(
        "--override_model",
        "-om",
        help="Override model to use for all tasks instead of task-specific models (e.g., qwen2.5:7b, qwen2.5:14b, qwen2.5:32b, gpt-oss:20b)",
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

    # Parse arguments
    args = parser.parse_args()

    # =================== Post-Processing & Validation ===================

    # Convert resume_dir to resume flag
    if args.resume_dir:
        args.resume = True

    # Set model name based on size for local backend
    if args.backend == "local":
        args.model_name = f"qwen2.5:{args.model_size}"

    # Validate method combinations
    if "storm" in args.methods and args.backend == "local":
        parser.error(
            "STORM method is not supported with --backend local. Use --backend ollama for STORM."
        )

    # Validate ollama-specific args
    if args.backend != "ollama" and args.ollama_host != "http://10.167.31.201:11434/":
        parser.error("--ollama_host can only be used with --backend ollama")

    # Validate local-specific args
    if args.backend != "local":
        if args.device != "auto":
            parser.error("--device can only be used with --backend local")
        if args.model_path:
            parser.error("--model_path can only be used with --backend local")
        if args.model_size != "32b":
            parser.error("--model_size can only be used with --backend local")

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
    if args.backend == "local" and "storm" in args.methods:
        print("Error: STORM is not supported with local backend")
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
        if args.backend == "ollama":
            print(f"  Ollama Host: {args.ollama_host}")
        else:
            print(f"  Device: {args.device}")
            print(f"  Model Size: {args.model_size}")
    else:
        print("❌ Argument validation failed")
        exit(1)
