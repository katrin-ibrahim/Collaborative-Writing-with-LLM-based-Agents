# FILE: cli_args.py for local models
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on 10 topics
  %(prog)s --num_topics 10

  # Run only direct prompting on 5 topics
  %(prog)s --methods direct --num_topics 5

  # Use custom model path
  %(prog)s --model_path /path/to/models --num_topics 5
        """,
    )

    parser.add_argument(
        "-p",
        "--model_path",
        default="models/",
        help="Path to local models directory",
    )
    parser.add_argument(
        "-s",
        "--model_size",
        choices=["7b", "14b", "32b"],
        default="32b",
        help="Which Qwen2.5 model size to use (7b, 14b, or 32b)",
    )
    parser.add_argument(
        "-n", "--num_topics", type=int, default=5, help="Number of topics to evaluate"
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=["direct", "storm"],
        choices=["direct", "storm"],
        help="Methods to run (direct, storm)",
    )
    parser.add_argument(
        "-c",
        "--model_config",
        default="config/models.yaml",
        help="Model configuration file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="results/local",
        help="Output directory for results",
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
    parser.add_argument(
        "--data_source",
        default="freshwiki",
        help="Data source: 'freshwiki' or path to file containing topics",
    )

    args = parser.parse_args()

    # Process model size to update default model in config
    args.model_name = f"qwen2.5:{args.model_size}"

    return args
