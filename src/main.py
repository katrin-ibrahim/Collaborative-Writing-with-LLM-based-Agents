#!/usr/bin/env python3
"""
Main runner for Ollama-based baseline experiments.
Clean architecture without HPC workarounds.
"""
import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import json
import logging
import time
from datetime import datetime

from utils.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from config.model_config import ModelConfig
from baselines.runner import BaselineRunner

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on 10 topics
  %(prog)s --num_topics 10
  
  # Run only STORM and Self-RAG on 100 topics
  %(prog)s --methods storm self_rag --num_topics 100
  
  # Use custom Ollama host
  %(prog)s --ollama_host http://localhost:11434/ --num_topics 5
        """
    )
    
    parser.add_argument(
        "--ollama_host",
        default="http://10.167.31.201:11434/",
        help="Ollama server URL"
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        default=5,
        help="Number of topics to evaluate"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["direct", "storm", "self_rag"],
        choices=["direct", "storm", "self_rag"],
        help="Methods to run"
    )
    parser.add_argument(
        "--model_config",
        default="config/models.yaml",
        help="Model configuration file"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip automatic evaluation"
    )
    parser.add_argument(
        "--output_dir",
        default="results/ollama",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


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
    """Main experiment runner."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("🦙 Ollama Baseline Experiment Runner")
    logger.info(f"📡 Ollama host: {args.ollama_host}")
    logger.info(f"📋 Methods: {', '.join(args.methods)}")
    logger.info(f"📝 Topics: {args.num_topics}")
    
    try:
        # Load model configuration
        model_config = load_model_config(args.model_config)
        logger.info(f"🤖 Models configured:")
        logger.info(f"  - Research: {model_config.research_model}")
        logger.info(f"  - Outline: {model_config.outline_model}")
        logger.info(f"  - Writing: {model_config.writing_model}")
        logger.info(f"  - Critique: {model_config.critique_model}")
        
        # Initialize runner
        runner = BaselineRunner(
            ollama_host=args.ollama_host,
            model_config=model_config
        )
        
        # Load FreshWiki topics
        logger.info("Loading FreshWiki topics...")
        freshwiki = FreshWikiLoader()
        entries = freshwiki.get_evaluation_sample(args.num_topics)
        
        if not entries:
            logger.error("No FreshWiki entries found!")
            return 1
        
        topics = [entry.topic for entry in entries]
        logger.info(f"✅ Loaded {len(topics)} topics")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run experiments
        logger.info("🚀 Starting experiments...")
        start_time = time.time()
        
        all_results = runner.run_all_baselines(topics, args.methods)
        
        total_time = time.time() - start_time
        logger.info(f"⏱️  Total experiment time: {total_time:.1f}s")
        
        # Evaluate if requested
        if not args.skip_evaluation:
            logger.info("📊 Evaluating results...")
            evaluator = ArticleEvaluator()
            
            for topic, methods_results in all_results.items():
                # Find corresponding reference
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
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert articles to dicts for JSON serialization
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
                        "research": model_config.research_model,
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
        
        # Print summary
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