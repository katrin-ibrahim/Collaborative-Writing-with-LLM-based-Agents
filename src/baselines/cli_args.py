# FILE: cli_args.py
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on 10 topics
  %(prog)s --num_topics 10

  # Run only STORM on 100 topics
  %(prog)s --methods storm --num_topics 100

  # Use custom Ollama host
  %(prog)s --ollama_host http://localhost:11434/ --num_topics 5
        """,
    )

    parser.add_argument(
        "-H",
        "--ollama_host",
        default="http://10.167.31.201:11434/",
        help="Ollama server URL",
    )
    parser.add_argument(
        "-n", "--num_topics", type=int, default=5, help="Number of topics to evaluate"
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=["direct", "storm", "rag"],
        choices=["direct", "storm", "rag"],
        help="Methods to run",
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
        default="results/ollama",
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
        "-r", "--resume_dir", type=str, help="Resume from specific run directory path"
    )

    return parser.parse_args()
