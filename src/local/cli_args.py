# FILE: cli_args.py for local models
import argparse
from datetime import datetime


def parse_arguments(default_methods=None):
    if default_methods is None:
        default_methods = ["direct_prompting"]
    
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run direct prompting on 10 topics
  %(prog)s --topic_limit 10

  # Use custom model path
  %(prog)s --model_path /path/to/models --topic_limit 5

  # Resume an experiment
  %(prog)s --resume --experiment_name my_experiment
        """,
    )

    parser.add_argument(
        "--model_path",
        default="models/",
        help="Path to local models directory",
    )
    parser.add_argument(
        "--model_size",
        choices=["3b", "32b", "72b"],
        default="32b",
        help="Which Qwen2.5 model size to use (3b, 32b, or 72b)",
    )
    parser.add_argument(
        "--topic_limit", 
        type=int, 
        default=5, 
        help="Number of topics to evaluate"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=default_methods,
        choices=["direct_prompting"],
        help="Methods to run (currently only direct_prompting is supported)",
    )
    parser.add_argument(
        "--model_config",
        default="config/models.yaml",
        help="Model configuration file",
    )
    parser.add_argument(
        "--output_dir",
        default="results/local",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves intermediate files)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing experiment",
    )
    parser.add_argument(
        "--experiment_name",
        default=f"local_direct_prompting_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Name for the experiment",
    )
    parser.add_argument(
        "--data_source",
        default="freshwiki",
        help="Data source: 'freshwiki' or path to file containing topics",
    )

    return parser.parse_args()
