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
    Merges with existing results.json if it exists to allow incremental runs.
    """
    results_file = output_dir / "results.json"
    articles_dir = output_dir / "articles"

    logger.info(f"Saving final results to: {results_file}")

    # Load existing results if they exist
    existing_data = {}
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                existing_data = json.load(f)
            logger.info("Loaded existing results.json for merging")
        except Exception as e:
            logger.warning(f"Could not load existing results.json: {e}")
            existing_data = {}

    # Build results structure (start with existing or new)
    data = (
        existing_data
        if existing_data
        else {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "backend": backend,
                "methods": {},
                "total_time": 0.0,
                "topics_processed": 0,
            },
            "results": {},
        }
    )

    # Update timestamp and add to total time
    data["summary"]["timestamp"] = datetime.now().isoformat()
    data["summary"]["total_time"] = data["summary"].get("total_time", 0.0) + total_time
    data["summary"]["backend"] = backend

    # Initialize all topics
    for topic in topics:
        if topic not in data["results"]:
            data["results"][topic] = {}

    # Scan articles directory for actual results
    if articles_dir.exists():
        for method in methods:
            method_articles = 0

            # Check for method_topic.md pattern
            for article_file in articles_dir.glob(f"{method}_*.md"):
                # Extract the filename without extension
                filename = article_file.stem

                # Check if this file actually belongs to a different method
                # that has this method as a prefix (e.g., writer vs writer_reviewer)
                # We need to ensure exact method match
                belongs_to_different_method = False
                for other_method in methods:
                    if other_method != method and other_method.startswith(method + "_"):
                        # Check if this file actually belongs to the longer method name
                        if filename.startswith(f"{other_method}_"):
                            belongs_to_different_method = True
                            break

                if belongs_to_different_method:
                    continue

                # Extract topic part after method name
                topic_part = filename[len(method) + 1 :]

                # Keep topic with underscores to match the topics list format
                if "and_or" in topic_part:
                    topic = topic_part.replace("and_or", "and/or")
                else:
                    topic = topic_part

                # Load metadata if available
                metadata_file = articles_dir / f"{article_file.stem}_metadata.json"
                generation_time = 0.0
                word_count = 0
                token_usage = None
                collaboration_metrics = None
                theory_of_mind = None

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)

                            # Verify the method recorded in the file matches the method we are processing.
                            # This is the only way to distinguish 'writer' from 'writer_reviewer'
                            # when running single-method experiments.
                            saved_method = metadata.get("method")
                            if saved_method and saved_method != method:
                                # Skip this file if it belongs to a different method (e.g. writer_reviewer)
                                continue

                            generation_time = metadata.get("generation_time", 0.0)
                            word_count = metadata.get("word_count", 0)
                            token_usage = metadata.get("token_usage")
                            collaboration_metrics = metadata.get(
                                "collaboration_metrics"
                            )
                            theory_of_mind = metadata.get("theory_of_mind")
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
                    except Exception:
                        word_count = 0

                # Ensure topic exists in results
                if topic not in data["results"]:
                    data["results"][topic] = {}

                method_data = {
                    "success": True,
                    "generation_time": generation_time,
                    "word_count": word_count,
                    "article_path": str(article_file.relative_to(output_dir)),
                }

                # Add metadata path for easy access by analysis scripts
                if metadata_file.exists():
                    method_data["metadata_path"] = str(
                        metadata_file.relative_to(output_dir)
                    )

                # Add token usage if available
                if token_usage:
                    method_data["token_usage"] = token_usage

                # Add collaboration metrics summary if available
                if collaboration_metrics:
                    method_data["collaboration_metrics"] = {
                        "total_feedback_items": collaboration_metrics.get(
                            "total_feedback_items", 0
                        ),
                        "feedback_resolution_rate": collaboration_metrics.get(
                            "feedback_resolution_rate", 0.0
                        ),
                        "convergence": collaboration_metrics.get("convergence", {}),
                    }

                # Add ToM metrics summary if available
                if theory_of_mind:
                    method_data["theory_of_mind"] = {
                        "total_predictions": theory_of_mind.get("total_predictions", 0),
                        "accuracy_rate": theory_of_mind.get("accuracy_rate", 0.0),
                    }

                data["results"][topic][method] = method_data
                method_articles += 1

            # Check for method/topic.md structure
            method_dir = articles_dir / method
            if method_dir.exists() and method_dir.is_dir():
                for article_file in method_dir.glob("*.md"):
                    topic = article_file.stem

                    if topic not in data["results"]:
                        data["results"][topic] = {}

                    data["results"][topic][method] = {
                        "success": True,
                        "article_path": str(article_file.relative_to(output_dir)),
                    }
                    method_articles += 1

            # Update method summary (merge with existing counts)
            existing_count = (
                data["summary"]["methods"].get(method, {}).get("article_count", 0)
            )
            data["summary"]["methods"][method] = {
                "article_count": max(method_articles, existing_count),
            }

    # Update topics_processed to count unique topics across all methods
    unique_topics = set()
    for topic, topic_data in data["results"].items():
        for method, method_data in topic_data.items():
            if method_data.get("success", False):
                unique_topics.add(topic)
    data["summary"]["topics_processed"] = len(unique_topics)

    # Mark missing results as failed (but don't overwrite existing success)
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
        logger.info(f"Final results saved to: {results_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")
        return False
