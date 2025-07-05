#!/usr/bin/env python3
"""
Main runner script for baseline evaluation.

Clean entry point for running baseline methods like STORM with local models.

Usage:
    python src/run.py --methods storm --num_topics 2
    python src/run.py --methods direct_prompting storm --num_topics 5
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from utils.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from config.storm_config import load_config
from baselines import BaselinesRunner

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --methods storm --num_topics 1
  %(prog)s --methods direct_prompting storm --num_topics 3
  %(prog)s --num_topics 5 --skip_evaluation
        """
    )
    
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Configuration file path (default: config.yaml)"
    )
    parser.add_argument(
        "--num_topics", 
        type=int, 
        default=2,
        help="Number of topics to evaluate (default: 2)"
    )
    parser.add_argument(
        "--methods", 
        nargs="+", 
        default=["direct_prompting", "storm"],
        choices=["direct_prompting", "storm"],
        help="Baseline methods to run (default: direct_prompting storm)"
    )
    parser.add_argument(
        "--skip_evaluation", 
        action="store_true",
        help="Skip automatic evaluation of generated articles"
    )
    parser.add_argument(
        "--log_level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def load_topics(num_topics):
    """Load topics from FreshWiki dataset."""
    logger.info("📚 Loading topics from FreshWiki...")
    
    freshwiki_loader = FreshWikiLoader()
    entries = freshwiki_loader.get_evaluation_sample(num_topics)
    
    if not entries:
        logger.error("❌ No FreshWiki entries found")
        return None, None
    
    logger.info(f"✅ Loaded {len(entries)} topics:")
    for i, entry in enumerate(entries, 1):
        word_count = len(entry.reference_content.split())
        section_count = len(entry.reference_outline)
        logger.info(f"  {i}. {entry.topic} ({word_count} words, {section_count} sections)")
    
    topics = [entry.topic for entry in entries]
    return topics, entries


def create_results_directory(methods, num_topics):
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    methods_str = "_".join(methods)
    results_dir = Path("results") / f"{methods_str}_{num_topics}topics_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Results directory: {results_dir}")
    return results_dir


def run_baselines(runner, topics, methods):
    """Run baseline methods on topics."""
    logger.info(f"🚀 Starting baseline evaluation...")
    logger.info(f"📋 Methods: {', '.join(methods)}")
    logger.info(f"📝 Topics: {len(topics)}")
    
    return runner.run_all_baselines(topics, methods)


def evaluate_results(all_results, entries, skip_evaluation):
    """Evaluate generated articles against reference."""
    if skip_evaluation:
        logger.info("⏭️  Skipping evaluation (--skip_evaluation flag)")
        return {}
    
    logger.info("📊 Evaluating generated articles...")
    evaluator = ArticleEvaluator()
    
    final_results = {}
    
    for topic, baseline_results in all_results.items():
        topic_result = {"baselines": {}}
        
        # Find corresponding reference entry
        entry = next((e for e in entries if e.topic == topic), None)
        if not entry:
            logger.warning(f"⚠️  No reference found for topic: {topic}")
        
        for method, result in baseline_results.items():
            method_result = {
                "generation_results": {
                    "success": result["success"],
                    "word_count": result["word_count"],
                    "metadata": result["article"].metadata
                },
                "evaluation_results": {}
            }
            
            # Run evaluation if possible
            if result["success"] and entry:
                try:
                    metrics = evaluator.evaluate_article(result["article"], entry)
                    method_result["evaluation_results"] = metrics
                    logger.info(f"✅ Evaluated {method} for {topic}")
                except Exception as e:
                    logger.warning(f"⚠️  Evaluation failed for {method} on {topic}: {e}")
            else:
                if not result["success"]:
                    logger.debug(f"Skipping evaluation for failed {method} on {topic}")
                if not entry:
                    logger.debug(f"Skipping evaluation for {topic} (no reference)")
            
            topic_result["baselines"][method] = method_result
        
        final_results[topic] = topic_result
    
    return final_results


def save_results(final_results, all_results, config, methods, results_dir):
    """Save results to JSON file."""
    results_file = results_dir / "results.json"
    
    # Prepare output data
    output_data = {
        "configuration": {
            "model_type": config.model_type,
            "local_model_path": config.local_model_path if config.model_type == "local" else None,
            "methods": methods,
            "storm_settings": {
                "max_conv_turn": config.max_conv_turn,
                "max_perspective": config.max_perspective,
                "search_top_k": config.search_top_k,
                "enable_polish": config.enable_polish,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature
            }
        },
        "results": final_results if final_results else all_results
    }
    
    # Save to file
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Results saved to: {results_file}")
    return results_file


def print_summary(final_results, all_results, methods):
    """Print evaluation summary."""
    results_to_use = final_results if final_results else all_results
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    for method in methods:
        successes = sum(
            1 for r in results_to_use.values() 
            if (r.get("baselines", {}).get(method, {}).get("generation_results", {}).get("success", False) if final_results 
                else r.get(method, {}).get("success", False))
        )
        
        total_words = sum(
            (r.get("baselines", {}).get(method, {}).get("generation_results", {}).get("word_count", 0) if final_results
             else r.get(method, {}).get("word_count", 0))
            for r in results_to_use.values()
            if (r.get("baselines", {}).get(method, {}).get("generation_results", {}).get("success", False) if final_results
                else r.get(method, {}).get("success", False))
        )
        
        avg_words = total_words / max(successes, 1)
        
        logger.info(f"📈 {method}: {successes}/{len(results_to_use)} successful, {avg_words:.0f} avg words")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Print header
    logger.info("🚀 Baseline Evaluation Runner")
    logger.info(f"📋 Methods: {', '.join(args.methods)}")
    logger.info(f"📝 Topics: {args.num_topics}")
    logger.info(f"📊 Evaluation: {'disabled' if args.skip_evaluation else 'enabled'}")
    
    try:
        # Load configuration
        logger.info("⚙️  Loading configuration...")
        config = load_config(args.config)
        logger.info(f"🤖 Model: {config.model_type}")
        if config.model_type == "local":
            logger.info(f"📁 Model path: {config.local_model_path}")
        
        # Load topics
        topics, entries = load_topics(args.num_topics)
        if not topics:
            return 1
        
        # Create results directory
        results_dir = create_results_directory(args.methods, args.num_topics)
        
        # Initialize runner
        logger.info("🔧 Initializing baseline runner...")
        runner = BaselinesRunner(config)
        
        # Run baselines
        all_results = run_baselines(runner, topics, args.methods)
        
        # Evaluate results
        final_results = evaluate_results(all_results, entries, args.skip_evaluation)
        
        # Save results
        save_results(final_results, all_results, config, args.methods, results_dir)
        
        # Print summary
        print_summary(final_results, all_results, args.methods)
        
        logger.info("✅ Evaluation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⛔ Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())