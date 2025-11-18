"""CLI for research visualizations."""

import argparse
import sys
from glob import glob
from pathlib import Path

import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.research_visualizer import ResearchVisualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate research visualizations for specific comparisons"
    )
    parser.add_argument(
        "--compare",
        choices=[
            "rm",
            "writing_mode",
            "revision_mode",
            "writer_sweep",
            "reviewer_sweep",
            "methods",
        ],
        required=True,
        help="Type of comparison to visualize",
    )

    parser.add_argument("--wiki", help="Path to wiki experiment (for rm comparison)")
    parser.add_argument("--faiss", help="Path to faiss experiment (for rm comparison)")

    parser.add_argument(
        "--section", help="Path to section experiment (for writing mode comparison)"
    )
    parser.add_argument(
        "--full", help="Path to full experiment (for writing mode comparison)"
    )

    parser.add_argument(
        "--pending", help="Path to pending experiment (for revision mode comparison)"
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Paths to experiments (for sweeps and methods). Supports wildcards.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        help="Filter to specific methods (e.g., writer writer_reviewer). For method comparison only.",
    )

    parser.add_argument(
        "--output",
        default="results/visualizations",
        help="Output directory (default: results/visualizations)",
    )

    return parser.parse_args()


def expand_wildcards(paths):
    """Expand wildcard paths."""
    expanded = []
    for path in paths:
        if "*" in path:
            expanded.extend(glob(path))
        else:
            expanded.append(path)
    return expanded


def main():
    args = parse_args()

    visualizer = ResearchVisualizer(output_base=args.output)

    try:
        if args.compare == "rm":
            if not args.wiki or not args.faiss:
                logger.error("RM comparison requires --wiki and --faiss")
                return 1
            visualizer.compare_rm(args.wiki, args.faiss)

        elif args.compare == "writing_mode":
            if not args.section or not args.full:
                logger.error("Writing mode comparison requires --section and --full")
                return 1
            visualizer.compare_writing_mode(args.section, args.full)

        elif args.compare == "revision_mode":
            if not args.pending or not args.section:
                logger.error(
                    "Revision mode comparison requires --pending and --section"
                )
                return 1
            visualizer.compare_revision_mode(args.pending, args.section)

        elif args.compare == "writer_sweep":
            if not args.experiments:
                logger.error("Writer sweep requires --experiments")
                return 1
            experiments = expand_wildcards(args.experiments)
            visualizer.compare_model_sweep(experiments, "writer_sweep")

        elif args.compare == "reviewer_sweep":
            if not args.experiments:
                logger.error("Reviewer sweep requires --experiments")
                return 1
            experiments = expand_wildcards(args.experiments)
            visualizer.compare_model_sweep(experiments, "reviewer_sweep")

        elif args.compare == "methods":
            if not args.experiments:
                logger.error("Method comparison requires --experiments")
                return 1
            experiments = expand_wildcards(args.experiments)
            visualizer.compare_methods(experiments, method_filter=args.methods)

        logger.info(f"Visualizations saved to: {args.output}")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
