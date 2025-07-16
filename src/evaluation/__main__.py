#!/usr/bin/env python3
"""
Main runner for evaluation module.
Evaluates articles from a results directory and adds evaluation metrics.
"""
import argparse
import sys
from pathlib import Path

import json
import logging
from typing import Dict

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from evaluation.evaluator import ArticleEvaluator
from utils.freshwiki_loader import FreshWikiLoader
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate articles from a results directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate articles in a specific results directory
  %(prog)s results/ollama/M=direct_N=1_T=d16.07_12:24

  # Evaluate with debug logging
  %(prog)s results/ollama/M=direct_N=1_T=d16.07_12:24 --log_level DEBUG
        """,
    )

    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the results directory containing articles and results.json",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-evaluation even if evaluation results already exist",
    )

    return parser.parse_args()


def load_results(results_dir: Path) -> Dict:
    """Load existing results from results.json."""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        return json.load(f)


def save_results(results_dir: Path, data: Dict):
    """Save updated results back to results.json."""
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("🔍 Article Evaluation Runner")

    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info(f"📂 Results directory: {results_dir}")

    # Load existing results
    try:
        data = load_results(results_dir)
        logger.info("📄 Loaded existing results")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return 1

    # Load FreshWiki dataset for evaluation
    logger.info("📚 Loading FreshWiki dataset...")
    freshwiki = FreshWikiLoader()
    entries = freshwiki.get_evaluation_sample(1000)  # Load all entries

    # Create evaluator
    evaluator = ArticleEvaluator()

    # Process each topic
    results = data.get("results", {})
    methods = data.get("configuration", {}).get("methods", [])

    evaluated_count = 0
    skipped_count = 0

    for topic, topic_results in results.items():
        # Find the corresponding FreshWiki entry
        entry = None
        for e in entries:
            if e.topic == topic:
                entry = e
                break

        if not entry:
            logger.warning(f"⚠️ No FreshWiki entry found for topic: {topic}")
            continue

        for method in methods:
            if method not in topic_results:
                continue

            method_result = topic_results[method]

            # Skip if not successful
            if not method_result.get("success"):
                logger.debug(f"⏭️ Skipping {method}/{topic} (not successful)")
                continue

            # Check if evaluation already exists
            if "evaluation" in method_result and not args.force:
                logger.debug(f"⏭️ Skipping {method}/{topic} (already evaluated)")
                skipped_count += 1
                continue

            # Run evaluation
            try:
                article = method_result["article"]
                eval_results = evaluator.evaluate_article(article, entry)
                method_result["evaluation"] = eval_results
                logger.info(f"✅ Evaluated {method} for {topic}")
                evaluated_count += 1
            except Exception as e:
                logger.error(f"❌ Evaluation failed for {method}/{topic}: {e}")
                method_result["evaluation_error"] = str(e)

    # Save updated results
    save_results(results_dir, data)

    # Summary
    logger.info(f"📊 Evaluation Summary:")
    logger.info(f"  - Evaluated: {evaluated_count} articles")
    logger.info(f"  - Skipped: {skipped_count} articles")
    logger.info(f"💾 Updated results saved to: {results_dir / 'results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
