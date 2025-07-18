#!/usr/bin/env python3
"""
Main runner for local model-based baseline experiments.
Uses locally hosted Qwen models instead of Ollama.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import json
import logging
import os
from typing import Dict, List

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from local.cli_args import parse_arguments
from local.runner import LocalBaselineRunner
from utils.freshwiki_loader import FreshWikiLoader
from utils.logging_setup import setup_logging
from utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


def merge_results_with_existing(
    existing_results: Dict,
    all_topics: List[str],
    direct_results: List,
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
    for i, result in enumerate(direct_results):
        if i < len(all_topics) and result is not None:
            topic = all_topics[i]
            all_results[topic]["direct_prompting"] = {
                "title": result.title,
                "word_count": result.word_count,
                "generation_time": result.generation_time,
                "timestamp": result.timestamp,
                "method": result.method,
                # Note: content is saved separately in articles/ folder
            }

    return {"results": all_results}


def filter_completed_work(
    existing_results: Dict, all_topics: List[str], methods: List[str]
) -> List[str]:
    """Filter out topics that have already been completed."""
    topics_to_run = []
    existing = existing_results.get("results", {})

    for topic in all_topics:
        if topic not in existing:
            topics_to_run.append(topic)
            continue

        topic_results = existing[topic]
        missing_methods = [method for method in methods if method not in topic_results]

        if missing_methods:
            topics_to_run.append(topic)

    logger.info(
        f"Found {len(topics_to_run)} topics to process out of {len(all_topics)} total"
    )
    return topics_to_run


def save_partial_results(
    results_file: Path,
    all_topics: List[str],
    direct_results: List,
    methods: List[str],
    existing_results: Dict = None,
):
    """Save partial results after each topic."""
    if existing_results is None:
        existing_results = {}

    results = merge_results_with_existing(
        existing_results, all_topics, direct_results, methods
    )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved partial results")


def main():
    setup_logging()
    args = parse_arguments(default_methods=["direct_prompting"])

    # Create output manager
    output_dir = Path(args.output_dir) / args.experiment_name
    output_manager = OutputManager(
        base_output_dir=str(output_dir),
        debug_mode=args.debug,
    )
    
    # Create results file path
    results_file = output_dir / "results.json"

    # Load existing results if resuming
    existing_results = {}
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            existing_results = json.load(f)
        logger.info(f"Resuming experiment with existing results")

    # Load data based on the source
    if args.data_source == "freshwiki":
        loader = FreshWikiLoader()
        entries = loader.get_evaluation_sample(args.topic_limit)
        all_topics = [entry.topic for entry in entries]
    else:
        # Load from file
        data_path = Path(args.data_source)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {args.data_source}")

        all_topics = []
        if data_path.suffix == ".json":
            with open(data_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_topics = data
                elif isinstance(data, dict) and "topics" in data:
                    all_topics = data["topics"]
        else:
            with open(data_path, "r") as f:
                all_topics = [line.strip() for line in f if line.strip()]

    if not all_topics:
        raise ValueError("No topics found in the data source")

    logger.info(f"Loaded {len(all_topics)} topics")

    # Filter topics that need to be processed
    topics_to_run = filter_completed_work(existing_results, all_topics, args.methods)

    if not topics_to_run:
        logger.info("All topics have been completed. Nothing to run.")
        return

    # Initialize runner with local models
    model_path = args.model_path or "models/"
    runner = LocalBaselineRunner(
        model_path=model_path,
        output_manager=output_manager,
    )

    # Initialize result lists
    direct_results = [None] * len(all_topics)

    # Process each topic
    for i, topic in enumerate(all_topics):
        if topic not in topics_to_run:
            # Load existing result
            if topic in existing_results.get("results", {}):
                existing_topic_results = existing_results["results"][topic]
                if "direct_prompting" in existing_topic_results:
                    from local.data_models import Article
                    result_data = existing_topic_results["direct_prompting"]
                    # For resume: create minimal Article object (content not needed for resume)
                    direct_results[i] = Article(
                        title=result_data["title"],
                        content="",  # Content is in articles/ folder, not needed for resume
                        word_count=result_data["word_count"],
                        generation_time=result_data["generation_time"],
                        timestamp=result_data["timestamp"],
                        method=result_data.get("method", "local_direct_prompting"),
                    )
            continue

        logger.info(f"Processing topic {i+1}/{len(all_topics)}: {topic}")

        # Run direct prompting
        if "direct_prompting" in args.methods:
            try:
                result = runner.run_direct_prompting(topic)
                direct_results[i] = result
                logger.info(
                    f"Direct prompting completed for '{topic}': "
                    f"{result.word_count} words in {result.generation_time:.2f}s"
                )
            except Exception as e:
                logger.error(f"Direct prompting failed for '{topic}': {e}")
                direct_results[i] = None

        # Save partial results after each topic
        save_partial_results(
            results_file, all_topics, direct_results, args.methods, existing_results
        )

    # Final results merge and save
    final_results = merge_results_with_existing(
        existing_results, all_topics, direct_results, args.methods
    )

    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    successful_direct = sum(1 for r in direct_results if r is not None)

    logger.info("\n" + "=" * 50)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Total topics: {len(all_topics)}")
    logger.info(f"Direct prompting successful: {successful_direct}/{len(all_topics)}")
    logger.info(f"Results saved to: {results_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
