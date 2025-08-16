#!/usr/bin/env python3
"""
Main runner for evaluation module.
Evaluates articles from a results directory and adds evaluation metrics.

Updated to use consolidated metrics functions instead of class-based approach.
"""
import argparse
import sys
import time
from pathlib import Path

import json
import logging
from typing import Dict, Optional

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.evaluation.evaluator import ArticleEvaluator
from src.utils.freshwiki_loader import FreshWikiLoader
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate articles from a results directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate articles in a specific results directory
  %(prog)s results/ollama/run_20250709_104159

  # Evaluate with debug logging
  %(prog)s results/ollama/run_20250709_104159 --log_level DEBUG
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
    """Load existing results from results.json, creating dummy structure if missing."""
    results_file = results_dir / "results.json"

    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}")
        logger.info("Creating dummy results.json from articles directory...")

        # Create dummy results by scanning articles directory
        dummy_results = create_dummy_results(results_dir)

        # Save the dummy results for future use
        try:
            with open(results_file, "w") as f:
                json.dump(dummy_results, f, indent=2)
            logger.info(f"✅ Created dummy results.json: {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save dummy results.json: {e}")

        return dummy_results

    with open(results_file, "r") as f:
        return json.load(f)


def create_dummy_results(results_dir: Path) -> Dict:
    """Create dummy results structure by scanning articles directory."""
    from datetime import datetime

    articles_dir = results_dir / "articles"

    # Detect methods and topics from articles directory
    methods = set()
    topics = set()

    if articles_dir.exists():
        # Scan for method_topic.md pattern
        for article_file in articles_dir.glob("*.md"):
            if "_metadata.json" in article_file.name:
                continue

            filename = article_file.stem

            # Try to split method from topic
            for potential_method in ["direct", "storm", "rag", "collaborative"]:
                if filename.startswith(f"{potential_method}_"):
                    methods.add(potential_method)
                    topic_part = filename[len(potential_method) + 1 :]

                    # Handle special cases
                    if "and_or" in topic_part:
                        topic = topic_part.replace("and_or", "and/or")
                    else:
                        topic = topic_part.replace("_", " ")

                    topics.add(topic)
                    break

        # Also check for method/topic.md structure
        for potential_method in ["direct", "storm", "rag", "collaborative"]:
            method_dir = articles_dir / potential_method
            if method_dir.exists() and method_dir.is_dir():
                methods.add(potential_method)
                for article_file in method_dir.glob("*.md"):
                    topic = article_file.stem.replace("_", " ")
                    topics.add(topic)

    methods = list(methods) if methods else ["direct", "storm", "rag"]
    topics = list(topics) if topics else ["dummy_topic"]

    logger.info(f"Detected methods: {methods}")
    logger.info(f"Detected topics: {topics}")

    # Create dummy structure
    dummy_data = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "backend": "unknown",
            "methods": {
                method: {"model": "unknown", "article_count": 0} for method in methods
            },
            "total_time": 0.0,
            "topics_processed": len(topics),
        },
        "results": {},
    }

    # Initialize all topic-method combinations
    for topic in topics:
        dummy_data["results"][topic] = {}
        for method in methods:
            # Check if article actually exists
            article_exists = False
            word_count = 0

            # Check method_topic.md pattern
            safe_topic = topic.replace(" ", "_").replace("/", "_")
            article_file = articles_dir / f"{method}_{safe_topic}.md"

            if article_file.exists():
                article_exists = True
                try:
                    with open(article_file, "r") as f:
                        content = f.read()
                    word_count = len(content.split())
                except:
                    word_count = 0
            else:
                # Check method/topic.md pattern
                method_dir = articles_dir / method
                if method_dir.exists():
                    topic_file = method_dir / f"{safe_topic}.md"
                    if topic_file.exists():
                        article_exists = True
                        try:
                            with open(topic_file, "r") as f:
                                content = f.read()
                            word_count = len(content.split())
                        except:
                            word_count = 0

            if article_exists:
                dummy_data["results"][topic][method] = {
                    "success": True,
                    "generation_time": 10.0,  # Dummy value
                    "word_count": word_count,
                    "article_path": f"articles/{method}_{safe_topic}.md",
                }
                dummy_data["summary"]["methods"][method]["article_count"] += 1
            else:
                dummy_data["results"][topic][method] = {
                    "success": False,
                    "error": "Article file not found",
                }

    return dummy_data


def save_results(results_dir: Path, data: Dict):
    """Save updated results back to results.json."""
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_article_content(results_dir: Path, method: str, topic: str) -> Optional[str]:
    """Get article content from file without storing in memory."""
    articles_dir = results_dir / "articles"

    # Normalize topic name (replace spaces with underscores for filenames)
    topic_normalized = topic.replace(" ", "_")
    topic_normalized = topic_normalized.replace("/", "_")  # Handle slashes too

    # Try different filename patterns
    possible_files = [
        articles_dir / f"{method}_{topic_normalized}.md",
        articles_dir / f"{method}_{topic}.md",  # Original with spaces
    ]

    for article_file in possible_files:
        if article_file.exists():
            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    logger.debug(f"Found article content at {article_file}")
                    return content
            except Exception as e:
                logger.warning(f"Failed to read {article_file}: {e}")
                continue

    logger.warning(f"No article content found for {method}/{topic}")
    logger.info(f"Tried files: {[str(f) for f in possible_files]}")
    return None


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info(f"Starting evaluation of results in: {results_dir}")
    start_time = time.time()

    try:
        # Load existing results
        logger.info("Loading existing results...")
        data = load_results(results_dir)

        # Check if evaluation already exists and not forcing
        if not args.force:
            results = data.get("results", {})
            has_evaluations = any(
                "evaluation" in method_data or "metrics" in method_data
                for topic_data in results.values()
                for method_data in topic_data.values()
            )
            if has_evaluations:
                logger.info(
                    "Evaluation results already exist. Use --force to re-evaluate."
                )
                return 0

        # Load FreshWiki dataset for evaluation
        logger.info("Loading FreshWiki dataset...")
        freshwiki = FreshWikiLoader()
        entries = freshwiki.load_topics(1000)  # Load all available entries

        if not entries:
            logger.error("No FreshWiki entries loaded")
            return 1

        logger.info(f"Loaded {len(entries)} FreshWiki entries")

        # Create topic lookup
        topic_lookup = {entry.topic: entry for entry in entries}

        # Create evaluator
        evaluator = ArticleEvaluator()

        # Process each topic and method
        results = data.get("results", {})
        total_evaluations = 0
        successful_evaluations = 0

        for topic, topic_data in results.items():
            logger.info(f"Processing topic: {topic}")

            # Find matching FreshWiki entry
            freshwiki_entry = None
            for entry in entries:
                # Normalize both titles by replacing underscores with spaces for comparison
                entry_topic_normalized = entry.topic.replace("_", " ").lower()
                topic_normalized = topic.replace("_", " ").lower()

                if (
                    entry_topic_normalized == topic_normalized
                    or topic_normalized in entry_topic_normalized
                    or entry_topic_normalized in topic_normalized
                ):
                    freshwiki_entry = entry
                    break

            if not freshwiki_entry:
                logger.warning(f"No FreshWiki entry found for topic: {topic}")
                continue

            for method, method_data in topic_data.items():
                if not isinstance(method_data, dict):
                    continue

                logger.debug(f"Evaluating {method} for {topic}")
                total_evaluations += 1

                # Skip if already evaluated and not forcing
                if not args.force and (
                    "evaluation" in method_data or "metrics" in method_data
                ):
                    logger.debug(f"Skipping {method}/{topic} - already evaluated")
                    successful_evaluations += 1
                    continue

                # Get article content
                article_content = None

                # Try to get content from stored path
                if "article_path" in method_data:
                    article_file = results_dir / method_data["article_path"]
                    if article_file.exists():
                        try:
                            with open(article_file, "r", encoding="utf-8") as f:
                                article_content = f.read().strip()
                        except Exception as e:
                            logger.warning(
                                f"Failed to read article from path {article_file}: {e}"
                            )

                # Fallback to search for article file
                if not article_content:
                    article_content = get_article_content(results_dir, method, topic)

                if not article_content:
                    logger.error(f"Could not find article content for {method}/{topic}")
                    # Add error to results
                    method_data["evaluation"] = {
                        metric: 0.0
                        for metric in [
                            "rouge_1",
                            "rouge_l",
                            "heading_soft_recall",
                            "heading_entity_recall",
                            "article_entity_recall",
                        ]
                    }
                    method_data["evaluation_error"] = "Could not find article content"
                    continue

                # Create Article object
                from src.utils.data_models import Article

                article = Article(title=topic, content=article_content)

                # Evaluate the article
                try:
                    logger.debug(f"Running evaluation for {method}/{topic}")
                    eval_results = evaluator.evaluate_article(article, freshwiki_entry)

                    if eval_results:
                        method_data["evaluation"] = eval_results
                        successful_evaluations += 1

                        # Ensure word count is present
                        if "word_count" not in method_data:
                            method_data["word_count"] = len(article_content.split())

                        logger.debug(f"✅ Evaluation completed for {method}/{topic}")
                        logger.debug(
                            f"   ROUGE-1: {eval_results.get('rouge_1', 0):.1f}%"
                        )
                        logger.debug(
                            f"   HSR: {eval_results.get('heading_soft_recall', 0):.1f}%"
                        )
                        logger.debug(
                            f"   HER: {eval_results.get('heading_entity_recall', 0):.1f}%"
                        )
                        logger.debug(
                            f"   AER: {eval_results.get('article_entity_recall', 0):.1f}%"
                        )
                    else:
                        logger.error(f"Evaluation returned None for {method}/{topic}")
                        method_data["evaluation"] = {
                            metric: 0.0
                            for metric in [
                                "rouge_1",
                                "rouge_l",
                                "heading_soft_recall",
                                "heading_entity_recall",
                                "article_entity_recall",
                            ]
                        }
                        method_data["evaluation_error"] = "Evaluation returned None"

                except Exception as e:
                    logger.error(f"Evaluation failed for {method}/{topic}: {e}")
                    method_data["evaluation"] = {
                        metric: 0.0
                        for metric in [
                            "rouge_1",
                            "rouge_l",
                            "heading_soft_recall",
                            "heading_entity_recall",
                            "article_entity_recall",
                        ]
                    }
                    method_data["evaluation_error"] = str(e)

        # Update results data with evaluation timestamp
        data["evaluation_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["evaluation_stats"] = {
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / total_evaluations
                if total_evaluations > 0
                else 0.0
            ),
        }

        # Save updated results
        logger.info("Saving updated results...")
        save_results(results_dir, data)

        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Evaluation completed in {elapsed_time:.1f}s")
        logger.info(
            f"📊 Evaluated {successful_evaluations}/{total_evaluations} articles successfully"
        )

        if successful_evaluations < total_evaluations:
            logger.warning(
                f"⚠️ {total_evaluations - successful_evaluations} evaluations failed"
            )

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
