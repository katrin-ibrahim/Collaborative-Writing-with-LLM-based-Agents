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
        """
    )

    parser.add_argument(
        "--ollama_host",
        default="http://10.167.31.201:11434/",
        help="Ollama server URL"
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        default=5,
        help="Number of topics to evaluate"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["direct", "storm"],
        choices=["direct", "storm"],
        help="Methods to run"
    )
    parser.add_argument(
        "--model_config",
        default="config/models.yaml",
        help="Model configuration file"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip automatic evaluation"
    )
    parser.add_argument(
        "--output_dir",
        default="results/ollama",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode (saves intermediate files)"
    )

    return parser.parse_args()
