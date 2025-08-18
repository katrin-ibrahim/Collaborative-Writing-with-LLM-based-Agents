"""
Experiment results management utilities.
Handles merging, saving, and serializing experimental results.
"""

from datetime import datetime
from pathlib import Path

import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def make_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if hasattr(obj, "__dict__"):
        return {k: make_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj


def merge_results_with_existing(
    existing_results: Dict,
    all_topics: List[str],
    direct_results: List,
    storm_results: List,
    rag_results: List,
    methods: List[str],
) -> Dict:
    """Merge new results with existing completed results."""

    # Start with existing results structure
    all_results = existing_results.get("results", {})

    # Ensure all topics have entries
    for topic in all_topics:
        if topic not in all_results:
            all_results[topic] = {}

    # Process direct results
    if "direct" in methods:
        # Add new direct results
        for result in direct_results:
            topic = result.title
            all_results[topic]["direct"] = {
                "success": True,
                "word_count": result.metadata.get("word_count", 0),
                "article": result,
            }

        # Ensure all topics have direct entries (mark missing as not found)
        for topic in all_topics:
            if "direct" not in all_results[topic]:
                # Check if it should be completed (this handles the case where
                # the topic was completed but not in our current batch)
                all_results[topic]["direct"] = {
                    "success": False,
                    "error": "Direct result not found in current batch",
                }

    # Process storm results
    if "storm" in methods:
        # Add new storm results
        for result in storm_results:
            topic = result.title
            all_results[topic]["storm"] = {
                "success": True,
                "word_count": result.metadata.get("word_count", 0),
                "article": result,
            }

        # Ensure all topics have storm entries
        for topic in all_topics:
            if "storm" not in all_results[topic]:
                all_results[topic]["storm"] = {
                    "success": False,
                    "error": "STORM result not found in current batch",
                }

    # Process rag results
    if "rag" in methods:
        # Add new rag results
        for result in rag_results:
            topic = result.title
            all_results[topic]["rag"] = {
                "success": True,
                "word_count": result.metadata.get("word_count", 0),
                "article": result,
            }

        # Ensure all topics have rag entries
        for topic in all_topics:
            if "rag" not in all_results[topic]:
                all_results[topic]["rag"] = {
                    "success": False,
                    "error": "RAG result not found in current batch",
                }

    return all_results


def save_final_results(
    output_dir: Path,
    topics: List[str],
    methods: List[str],
    total_time: float,
    backend: str = "unknown",
) -> bool:
    """
    Generate and save final results.json file from completed experiment.

    Scans the articles directory to create a complete results structure.
    """
    results_file = output_dir / "results.json"
    articles_dir = output_dir / "articles"

    logger.info(f"Saving final results to: {results_file}")

    # Build results structure
    data = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "backend": backend,
            "methods": {},
            "total_time": total_time,
            "topics_processed": len(topics),
        },
        "results": {},
    }

    # Initialize all topics
    for topic in topics:
        data["results"][topic] = {}

    # Scan articles directory for actual results
    if articles_dir.exists():
        for method in methods:
            method_articles = 0

            # Check for method_topic.md pattern
            for article_file in articles_dir.glob(f"{method}_*.md"):
                topic_part = article_file.stem[len(method) + 1 :]

                # Handle topics with slashes that were replaced with underscores
                if "and_or" in topic_part:
                    topic = topic_part.replace("and_or", "and/or")
                else:
                    topic = topic_part.replace("_", " ")

                # Load metadata if available
                metadata_file = articles_dir / f"{article_file.stem}_metadata.json"
                generation_time = 0.0
                word_count = 0
                model_info = "unknown"

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            generation_time = metadata.get("generation_time", 0.0)
                            word_count = metadata.get("word_count", 0)
                            model_info = metadata.get("model", "unknown")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata for {article_file}: {e}"
                        )

                # Calculate word count from content if not in metadata
                if word_count == 0:
                    try:
                        with open(article_file, "r") as f:
                            content = f.read()
                        word_count = len(content.split())
                    except:
                        word_count = 0

                # Ensure topic exists in results
                if topic not in data["results"]:
                    data["results"][topic] = {}

                data["results"][topic][method] = {
                    "success": True,
                    "generation_time": generation_time,
                    "word_count": word_count,
                    "article_path": str(article_file.relative_to(output_dir)),
                }
                method_articles += 1

            # Check for method/topic.md structure
            method_dir = articles_dir / method
            if method_dir.exists() and method_dir.is_dir():
                for article_file in method_dir.glob("*.md"):
                    topic = article_file.stem.replace("_", " ")

                    if topic not in data["results"]:
                        data["results"][topic] = {}

                    data["results"][topic][method] = {
                        "success": True,
                        "article_path": str(article_file.relative_to(output_dir)),
                    }
                    method_articles += 1

            # Update method summary
            data["summary"]["methods"][method] = {
                "model": model_info,
                "article_count": method_articles,
            }

    # Mark missing results as failed
    for topic in topics:
        for method in methods:
            if method not in data["results"][topic]:
                data["results"][topic][method] = {
                    "success": False,
                    "error": f"No {method} result found for topic",
                }

    try:
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"✅ Final results saved to: {results_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")
        return False
