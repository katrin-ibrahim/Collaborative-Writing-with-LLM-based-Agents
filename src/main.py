#!/usr/bin/env python3
"""
Main runner for Ollama-based baseline experiments.
Clean architecture without HPC workarounds.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import json
import logging
import os
from typing import Dict, List

from utils.experiment_state_manager import ExperimentStateManager

# Add src directory to path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from baselines.runner import BaselineRunner
from cli_args import parse_arguments  # Import from new file
from config.model_config import ModelConfig
from evaluation.evaluator import ArticleEvaluator
from utils.freshwiki_loader import FreshWikiLoader
from utils.logging_setup import setup_logging
from utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


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


def setup_output_directory(args) -> Path:
    """Setup output directory for new or resumed experiments."""
    base_output_dir = Path(args.output_dir)

    if args.resume_dir:
        # Resume from specific directory
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            raise ValueError(f"Resume directory does not exist: {output_dir}")
        logger.info(f"📂 Resuming from specified directory: {output_dir}")
        return output_dir

    # If no resume_dir specified, create new experiment directory

    # Create new run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Created new run directory: {output_dir}")
    return output_dir


def load_model_config(config_file: str) -> ModelConfig:
    """Load model configuration from file or use defaults."""
    if os.path.exists(config_file):
        try:
            import yaml

            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
            return ModelConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")

    logger.info("Using default model configuration")
    return ModelConfig()


def main():
    args = parse_arguments()

    setup_logging(args.log_level)
    if args.log_level == "DEBUG":
        logging.getLogger("evaluation.metrics.entity_metrics").setLevel(logging.DEBUG)
        logging.getLogger("evaluation.evaluator").setLevel(logging.DEBUG)
    else:
        logging.getLogger("evaluation.metrics.entity_metrics").setLevel(logging.INFO)
        logging.getLogger("evaluation.evaluator").setLevel(logging.INFO)

    logger.info("🦙 Ollama Baseline Experiment Runner")
    logger.info(f"📡 Ollama host: {args.ollama_host}")
    logger.info(f"📝 Topics: {args.num_topics}")

    try:
        model_config = load_model_config(args.model_config)

        logger.info(f"🤖 Models configured:")
        logger.info(f"  - Research: {model_config.default_model}")
        logger.info(f"  - Outline: {model_config.outline_model}")
        logger.info(f"  - Writing: {model_config.writing_model}")
        logger.info(f"  - Critique: {model_config.critique_model}")

        # Handle resume logic
        output_dir = setup_output_directory(args)
        output_manager = OutputManager(str(output_dir), debug_mode=args.debug)

        # Initialize state manager
        state_manager = ExperimentStateManager(output_dir, args.methods)

        # Load checkpoint if resuming
        is_resume = state_manager.load_checkpoint()
        if is_resume:
            logger.info("🔄 Resuming experiment from checkpoint")
        else:
            logger.info("🆕 Starting new experiment")

        runner = BaselineRunner(
            ollama_host=args.ollama_host,
            model_config=model_config,
            output_manager=output_manager,
        )
        runner.set_state_manager(state_manager)

        freshwiki = FreshWikiLoader()
        entries = freshwiki.get_evaluation_sample(args.num_topics)

        if not entries:
            logger.error("No FreshWiki entries found!")
            return 1

        topics = [entry.topic for entry in entries]
        logger.info(f"✅ Loaded {len(topics)} topics")

        # Analyze existing state and cleanup in-progress topics
        state_manager.analyze_existing_state(topics)
        state_manager.cleanup_and_restart_in_progress(topics)

        # Load existing results to merge with new ones
        existing_results = state_manager.load_existing_results()

        # Log progress summary
        progress_summary = state_manager.get_progress_summary(len(topics))
        logger.info("📊 Progress Summary:")
        for method, stats in progress_summary.items():
            if method in args.methods:
                logger.info(
                    f"  {method}: {stats['completed']}/{stats['total']} completed ({stats['progress_pct']}%)"
                )

        logger.info("🚀 Starting processing...")
        start_time = time.time()
        direct_results = []
        storm_results = []
        rag_results = []

        # Process methods with proper topic filtering
        if "direct" in args.methods:
            if len(topics) > 1:
                direct_results = runner.run_direct_batch(topics)
            else:
                remaining_direct = runner.filter_completed_topics(topics, "direct")
                if remaining_direct:
                    direct_results = [runner.run_direct_prompting(remaining_direct[0])]

        if "storm" in args.methods:
            if len(topics) > 1:
                storm_results = runner.run_storm_batch(topics)
            else:
                remaining_storm = runner.filter_completed_topics(topics, "storm")
                if remaining_storm:
                    storm_results = [runner.run_storm(remaining_storm[0])]

        if "rag" in args.methods:
            if len(topics) > 1:
                rag_results = runner.run_rag_batch(topics)
            else:
                remaining_rag = runner.filter_completed_topics(topics, "rag")
                if remaining_rag:
                    rag_results = [runner.run_rag(remaining_rag[0])]

        if not any(method in args.methods for method in ["direct", "storm", "rag"]):
            logger.error(
                "No valid methods specified. Please choose from: direct, storm, rag."
            )
            return 1
        # Merge results: combine existing completed results with new results
        all_results = merge_results_with_existing(
            existing_results,
            topics,
            direct_results,
            storm_results,
            rag_results,
            args.methods,
        )

        total_time = time.time() - start_time
        logger.info(f"⏱️ Total time: {total_time:.1f}s")

        # Run evaluation if not skipped (EVALUATION SECTION PRESERVED)
        if not args.skip_evaluation:
            logger.info("🔍 Starting evaluation...")
            evaluator = ArticleEvaluator()

            for topic, results in all_results.items():
                entry = next(e for e in entries if e.topic == topic)

                for method in args.methods:
                    if method in results and results[method].get("success"):
                        try:
                            article = results[method]["article"]
                            eval_results = evaluator.evaluate_article(article, entry)
                            results[method]["evaluation"] = eval_results
                            logger.info(f"✅ Evaluated {method} for {topic}")
                        except Exception as e:
                            logger.error(
                                f"❌ Evaluation failed for {method}/{topic}: {e}"
                            )
                            results[method]["evaluation_error"] = str(e)

        # Convert articles to serializable format for saving
        def make_serializable(obj):
            if hasattr(obj, "__dict__"):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        serializable_results = make_serializable(all_results)

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "configuration": {
                        "ollama_host": args.ollama_host,
                        "methods": args.methods,
                        "num_topics": args.num_topics,
                        "models": {
                            "default": model_config.default_model,
                            "outline": model_config.outline_model,
                            "writing": model_config.writing_model,
                            "critique": model_config.critique_model,
                            "retrieval": model_config.retrieval_model,
                        },
                    },
                    "results": serializable_results,
                    "summary": {
                        "total_time": total_time,
                        "topics_processed": len(topics),
                        "methods_run": args.methods,
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"💾 Results saved to: {results_file}")

        # SUMMARY SECTION PRESERVED
        logger.info("\n📊 SUMMARY")
        logger.info("=" * 50)
        for method in args.methods:
            successes = sum(
                1
                for r in all_results.values()
                if r.get(method, {}).get("success", False)
            )
            total_words = sum(
                r.get(method, {}).get("word_count", 0)
                for r in all_results.values()
                if r.get(method, {}).get("success", False)
            )
            avg_words = total_words / max(successes, 1)

            logger.info(
                f"{method}: {successes}/{len(topics)} successful, {avg_words:.0f} avg words"
            )

        logger.info("\n✅ Experiment completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("⛔ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
