#!/usr/bin/env python3
"""
Base runner implementation for baseline experiments.
This provides common functionality used by both Ollama and local baseline runners.
"""
import time
from datetime import datetime
from pathlib import Path

import json
import logging
from typing import Any

from src.config.baselines_model_config import ModelConfig
from src.utils.baselines_utils import (
    make_serializable,
    merge_results_with_existing,
    setup_output_directory,
)
from src.utils.experiment_state_manager import ExperimentStateManager
from src.utils.freshwiki_loader import FreshWikiLoader
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


def load_model_config(config_file: str) -> ModelConfig:
    """Load model configuration from file or use defaults."""
    if Path(config_file).exists():
        try:
            import yaml

            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
            return ModelConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")

    logger.info("Using default model configuration")
    return ModelConfig()


def run_baseline_experiment(args: Any, runner_class: Any, runner_name: str):
    """
    Run a baseline experiment using the provided arguments and runner class.

    Args:
        args: Command-line arguments from argparse
        runner_class: Class to instantiate for running the baseline methods
        runner_name: Name of the runner (for logging purposes)

    Returns:
        0 for success, 1 for failure
    """
    from src.utils.logging_setup import setup_logging

    setup_logging(args.log_level)

    logger.info(f"🔬 {runner_name} Baseline Experiment Runner")
    logger.info(f"📝 Topics: {args.num_topics}")

    try:
        model_config = load_model_config(args.model_config)

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

        # Create appropriate runner instance based on the runner class passed in
        if runner_name.lower() == "ollama":
            runner = runner_class(
                ollama_host=args.ollama_host,
                model_config=model_config,
                output_manager=output_manager,
            )
        else:  # Local runner
            runner = runner_class(
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

        if "storm" in args.methods and hasattr(runner, "run_storm_batch"):
            if len(topics) > 1:
                storm_results = runner.run_storm_batch(topics)
            else:
                remaining_storm = runner.filter_completed_topics(topics, "storm")
                if remaining_storm:
                    storm_results = [runner.run_storm(remaining_storm[0])]

        if "rag" in args.methods and hasattr(runner, "run_rag_batch"):
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

        serializable_results = make_serializable(all_results)

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "configuration": {
                        "runner_type": runner_name,
                        "methods": args.methods,
                        "num_topics": args.num_topics,
                        "models": {
                            "default": model_config.default_model,
                            "outline": model_config.outline_model,
                            "writing": model_config.writing_model,
                            "critique": model_config.critique_model,
                        },
                        **(
                            {"ollama_host": args.ollama_host}
                            if hasattr(args, "ollama_host")
                            else {}
                        ),
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
