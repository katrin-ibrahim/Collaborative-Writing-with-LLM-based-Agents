#!/usr/bin/env python3
"""
Cross-experiment analysis command-line interface.
Compare methods across multiple experimental runs.
"""

import argparse
import sys
from pathlib import Path

import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cross_experiment_visualizer import CrossExperimentVisualizer


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("cross_experiment_analysis.log"),
        ],
    )


def find_experiment_directories(base_path: str) -> list:
    """Find experiment directories that contain results.json files."""
    base = Path(base_path)
    experiment_dirs = []

    for item in base.iterdir():
        if item.is_dir():
            results_file = item / "results.json"
            if results_file.exists():
                experiment_dirs.append(str(item))

    return sorted(experiment_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-experiment comparison visualizations"
    )

    parser.add_argument(
        "experiments",
        nargs="+",
        help="Paths to experiment directories (or parent directory to auto-discover)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="*",
        help="Custom labels for experiments (must match number of experiment paths)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/cross_experiment_analysis",
        help="Output directory for charts (default: results/cross_experiment_analysis)",
    )

    parser.add_argument(
        "--method",
        "-m",
        help="Generate chart for specific method only (e.g., 'rag', 'storm')",
    )

    parser.add_argument(
        "--auto-discover",
        "-a",
        action="store_true",
        help="Auto-discover experiment directories from provided path(s)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    # Validate labels if provided
    if args.labels:
        if len(args.labels) != len(args.experiments):
            print(
                f"Error: Number of labels ({len(args.labels)}) must match number of experiments ({len(args.experiments)})"
            )
            return 1
    logger = logging.getLogger(__name__)

    # Handle experiment paths
    experiment_paths = []

    if args.auto_discover:
        # Auto-discover from provided paths
        for path in args.experiments:
            discovered = find_experiment_directories(path)
            experiment_paths.extend(discovered)
            logger.info(f"Discovered {len(discovered)} experiments in {path}")
    else:
        # Use provided paths directly
        experiment_paths = args.experiments

    if not experiment_paths:
        logger.error("No experiment directories found")
        return 1

    logger.info(f"Analyzing {len(experiment_paths)} experiments:")
    for path in experiment_paths:
        logger.info(f"  - {path}")

    # Create visualizer
    try:
        visualizer = CrossExperimentVisualizer(
            experiment_paths=experiment_paths,
            output_dir=args.output_dir,
            custom_labels=args.labels,
        )

        # Show experiment info
        info = visualizer.get_experiment_info()
        logger.info(f"Successfully loaded {info['total_experiments']} experiments")

        for exp_name, details in info["experiment_details"].items():
            logger.info(
                f"  {exp_name}: {details['total_topics']} topics, "
                f"methods: {', '.join(details['methods'])}"
            )

        # Generate charts
        if args.method:
            # Single method
            chart_path = visualizer.create_method_comparison_chart(args.method)
            if chart_path:
                logger.info(f"Generated chart for {args.method}: {chart_path}")
            else:
                logger.error(f"Failed to generate chart for method: {args.method}")
                return 1
        else:
            # All charts
            chart_paths = visualizer.generate_all_charts()

            if chart_paths:
                logger.info(f"Generated {len(chart_paths)} charts:")
                for chart_name, chart_path in chart_paths.items():
                    logger.info(f"  {chart_name}: {chart_path}")
            else:
                logger.error("Failed to generate any charts")
                return 1

        logger.info(f"Analysis complete. Charts saved to: {args.output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
