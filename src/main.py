#!/usr/bin/env python3
"""
Main runner for Ollama-based baseline experiments.
Clean architecture without HPC workarounds.
"""
import sys
import os
from pathlib import Path
import json
import logging
import time
from datetime import datetime

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from config.model_config import ModelConfig
from baselines.runner import BaselineRunner
from utils.output_manager import OutputManager
from cli_args import parse_arguments  # Import from new file

logger = logging.getLogger(__name__)


def load_model_config(config_file: str) -> ModelConfig:
    """Load model configuration from file or use defaults."""
    if os.path.exists(config_file):
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            return ModelConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")

    logger.info("Using default model configuration")
    return ModelConfig()


def main():
    args = parse_arguments()

    setup_logging(args.log_level)

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_manager = OutputManager(str(output_dir), debug_mode=args.debug)
    

        runner = BaselineRunner(
            ollama_host=args.ollama_host,
            model_config=model_config,
            output_manager=output_manager
        )

        freshwiki = FreshWikiLoader()
        entries = freshwiki.get_evaluation_sample(args.num_topics)

        if not entries:
            logger.error("No FreshWiki entries found!")
            return 1

        topics = [entry.topic for entry in entries]
        logger.info(f"✅ Loaded {len(topics)} topics")

      

        logger.info("🚀 Starting...")
        start_time = time.time()
        direct_results = []
        storm_results = []

        if "direct" in args.methods:
            if len(topics) > 1:
                direct_results = runner.run_direct_batch(topics)
            else:
                direct_results = [runner.run_direct_prompting(topics[0])]

        if "storm" in args.methods:
            if len(topics) > 1:
                storm_results = runner.run_storm_batch(topics)
            else:
                storm_results = [runner.run_storm(topics[0])]

        all_results = {}
        for entry in entries:
            topic = entry.topic
            all_results[topic] = {}

            if "direct" in args.methods:
                direct_result = next(
                    (r for r in direct_results if r.title == topic), None
                )
                if direct_result:
                    all_results[topic]["direct"] = {
                        "success": True,
                        "word_count": direct_result.metadata.get("word_count", 0),
                        "article": direct_result
                    }
                else:
                    all_results[topic]["direct"] = {
                        "success": False,
                        "error": "Direct result not found"
                    }

            if "storm" in args.methods:
                storm_result = next(
                    (r for r in storm_results if r.title == topic), None
                )
                if storm_result:
                    all_results[topic]["storm"] = {
                        "success": True,
                        "word_count": storm_result.metadata.get("word_count", 0),
                        "article": storm_result
                    }
                else:
                    all_results[topic]["storm"] = {
                        "success": False,
                        "error": "Storm result not found"
                    }
       
        total_time = time.time() - start_time
        logger.info(f"⏱️  Total experiment time: {total_time:.1f}s")

        if not args.skip_evaluation:
            logger.info("📊 Evaluating results...")
            evaluator = ArticleEvaluator()

            for topic, methods_results in all_results.items():
                entry = next((e for e in entries if e.topic == topic), None)
                if not entry:
                    continue

                for method, result in methods_results.items():
                    if result["success"]:
                        try:
                            metrics = evaluator.evaluate_article(
                                result["article"], entry
                            )
                            result["metrics"] = metrics
                        except Exception as e:
                            logger.warning(f"Evaluation failed for {method} on {topic}: {e}")

        results_file = output_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            serializable_results = {}
            for topic, methods_results in all_results.items():
                serializable_results[topic] = {}
                for method, result in methods_results.items():
                    serializable_results[topic][method] = {
                        "success": result["success"],
                        "word_count": result["word_count"],
                        "article": result["article"].to_dict(),
                        "metrics": result.get("metrics", {})
                    }

            json.dump({
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
                        "generation": model_config.generation_model,
                        "reflection": model_config.reflection_model
                    }
                },
                "results": serializable_results,
                "summary": {
                    "total_time": total_time,
                    "topics_processed": len(topics),
                    "methods_run": args.methods
                }
            }, f, indent=2)

        logger.info(f"💾 Results saved to: {results_file}")

        logger.info("\n📊 SUMMARY")
        logger.info("=" * 50)
        for method in args.methods:
            successes = sum(
                1 for r in all_results.values()
                if r.get(method, {}).get("success", False)
            )
            total_words = sum(
                r.get(method, {}).get("word_count", 0)
                for r in all_results.values()
                if r.get(method, {}).get("success", False)
            )
            avg_words = total_words / max(successes, 1)

            logger.info(f"{method}: {successes}/{len(topics)} successful, {avg_words:.0f} avg words")

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
