import argparse
import logging
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Import necessary modules
from utils.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from utils.data_models import Article
from config.storm_config import load_config, get_api_key

# Import STORM components
from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import DuckDuckGoSearchRM
import litellm

def initialize_storm(config) -> Optional[object]:
    """Initialize STORM with simple config"""
    logger = logging.getLogger(__name__)
    try:
        # Disable LiteLLM caching to avoid annotation errors
        litellm.cache = None
        os.environ["LITELLM_DISABLE_CACHE"] = "True"

        logger.info(f"Using {config.provider} provider with model: {config.model}")

        # Get API key
        api_key = get_api_key(config.provider)

        # Use STORM's LM configs
        lm_config = STORMWikiLMConfigs()

        # Create LiteLLM instance
        model_name = f"{config.provider}/{config.model}" if config.provider != "openai" else config.model

        llm = LitellmModel(
            model=model_name,
            max_tokens=50,
            api_key=api_key,
            temperature=0.7,
            timeout=30
        )

        # Set all STORM LM components
        lm_config.set_conv_simulator_lm(llm)
        lm_config.set_question_asker_lm(llm)
        lm_config.set_outline_gen_lm(llm)
        lm_config.set_article_gen_lm(llm)
        lm_config.set_article_polish_lm(llm)

        # Use DuckDuckGo search
        rm = DuckDuckGoSearchRM(k=config.search_top_k)

        # Create STORM runner arguments
        engine_args = STORMWikiRunnerArguments(
            output_dir="./storm_output",
            max_conv_turn=config.max_conv_turn,
            max_perspective=config.max_perspective,
            search_top_k=config.search_top_k,
            max_thread_num=config.max_thread_num
        )

        # Create STORM runner
        storm_runner = STORMWikiRunner(engine_args, lm_config, rm)

        return storm_runner

    except ImportError as e:
        logger.error(
            "STORM package not installed. Install with: pip install knowledge-storm==1.1.0 litellm duckduckgo-search")
        return None
    except Exception as e:
        logger.error(f"STORM initialization failed: {e}")
        return None


def run_storm_on_topic(storm_runner, topic: str, output_dir: Path, config) -> Tuple[Article, Optional[str]]:
    """Run STORM pipeline on topic"""

    logger = logging.getLogger(__name__)

    for attempt in range(config.max_retries):
        try:
            topic_dir = output_dir / topic.replace(" ", "_").replace("/", "_")
            topic_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Running STORM (attempt {attempt + 1}/{config.max_retries})...")

            # Add delay to reduce rate limiting
            if attempt > 0:
                delay = 10 + random.uniform(5, 15)
                logger.info(f"Adding {delay:.1f}s delay...")
                time.sleep(delay)
            elif attempt == 0:
                delay = random.uniform(2, 5)
                time.sleep(delay)

            # Run STORM pipeline
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=config.enable_polish
            )

            # Read generated article
            storm_output_dir = Path(storm_runner.args.output_dir)
            topic_subdir = storm_output_dir / topic.replace(" ", "_").replace("/", "_")

            article_files = [
                topic_subdir / "storm_gen_article_polished.txt",
                topic_subdir / "storm_gen_article.txt"
            ]

            content = None
            for article_file in article_files:
                if article_file.exists():
                    content = article_file.read_text(encoding='utf-8')
                    logger.info(f"✓ Read article from {article_file.name}")
                    break

            if not content:
                if topic_subdir.exists():
                    files = list(topic_subdir.glob("*.txt"))
                    if files:
                        content = files[0].read_text(encoding='utf-8')
                        logger.info(f"✓ Read article from {files[0].name}")

            if not content:
                content = f"# {topic}\n\nSTORM completed but no article content found"

            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm_full",
                    "word_count": len(content.split()),
                    "output_dir": str(topic_subdir),
                    "attempts": attempt + 1
                }
            ), None

        except Exception as e:
            error_msg = str(e)

            if "DuckDuckGoSearchException" in error_msg or "Ratelimit" in error_msg:
                logger.warning(f"Search rate limiting detected on attempt {attempt + 1}")
                error_msg = "DuckDuckGo search rate limiting"

            logger.warning(f"STORM attempt {attempt + 1} failed: {error_msg}")

            if attempt < config.max_retries - 1:
                base_wait = 10 if "rate" in error_msg.lower() else 5
                wait_time = base_wait * (2 ** attempt) + random.uniform(5, 15)
                logger.info(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                return Article(
                    title=topic,
                    content=f"# {topic}\n\nSTORM Error after {config.max_retries} attempts: {error_msg}",
                    sections={},
                    metadata={"error": error_msg, "attempts": config.max_retries}
                ), error_msg

    # This should never be reached, but add for completeness
    return Article(
        title=topic,
        content=f"# {topic}\n\nUnexpected error: max retries exceeded",
        sections={},
        metadata={"error": "max_retries_exceeded", "attempts": config.max_retries}
    ), "max_retries_exceeded"


def main():
    parser = argparse.ArgumentParser(description="Simplified STORM Baselines")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--num_topics", type=int, default=5)
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--delay_between_topics", type=float, default=30.0)

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Running STORM Baselines")

    # Load config
    config = load_config(args.config)

    logger.info(f"Provider: {config.provider}")
    logger.info(f"Model: {config.model}")
    logger.info(
        f"STORM settings: conv_turns={config.max_conv_turn}, perspectives={config.max_perspective}, retries={config.max_retries}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path("results") / f"storm_{args.num_topics}topics_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize STORM
    storm_runner = initialize_storm(config)
    if not storm_runner:
        logger.error("Failed to initialize STORM")
        return

    # Get topics from FreshWiki
    logger.info("Loading topics from FreshWiki...")
    freshwiki_loader = FreshWikiLoader()
    entries = freshwiki_loader.get_evaluation_sample(args.num_topics)

    if not entries:
        logger.error("No FreshWiki entries found")
        return

    logger.info(f"Loaded {len(entries)} topics")

    # Initialize evaluator
    evaluator = None
    if not args.skip_evaluation:
        evaluator = ArticleEvaluator()
        logger.info("Evaluation enabled")

    # Create STORM output directory
    storm_output_dir = results_dir / "storm_outputs"
    storm_output_dir.mkdir(exist_ok=True)

    # Process topics
    all_results = {}

    for i, entry in enumerate(entries, 1):
        topic = entry.topic
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {i}/{len(entries)}: {topic}")
        logger.info(f"{'=' * 60}")

        article, error = run_storm_on_topic(storm_runner, topic, storm_output_dir, config)

        topic_results = {
            "generation_results": {
                "success": error is None,
                "word_count": article.metadata.get("word_count", 0),
                "attempts": article.metadata.get("attempts", 1),
                "output_dir": article.metadata.get("output_dir"),
                "error": error
            },
            "evaluation_results": {}
        }

        # Evaluate if evaluator is available and no error occurred
        if evaluator and error is None:
            metrics = evaluator.evaluate_article(article, entry)
            topic_results["evaluation_results"] = metrics
            logger.info(f"STORM: {article.metadata.get('word_count', 0)} words")
        elif error:
            logger.error(f"STORM failed: {error}")

        all_results[topic] = topic_results

        # Add delay between topics
        if i < len(entries):
            jitter = random.uniform(0.8, 1.2)
            delay = args.delay_between_topics * jitter
            logger.info(f"Waiting {delay:.1f}s before next topic...")
            time.sleep(delay)

    # Save results
    results_file = results_dir / "results.json"
    final_results = {
        "configuration": {
            "provider": config.provider,
            "model": config.model,
            "storm_settings": {
                "max_conv_turn": config.max_conv_turn,
                "max_perspective": config.max_perspective,
                "search_top_k": config.search_top_k,
                "enable_polish": config.enable_polish,
                "max_retries": config.max_retries
            }
        },
        "results": all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    successes = sum(1 for r in all_results.values()
                    if r["generation_results"].get("success", False))
    total_words = sum(r["generation_results"].get("word_count", 0)
                      for r in all_results.values()
                      if r["generation_results"].get("success", False))
    avg_words = total_words / max(successes, 1)

    logger.info(f"STORM: {successes}/{len(all_results)} successful")
    logger.info(f"Average words: {avg_words:.0f}")

    if evaluator and successes > 0:
        rouge_scores = [r["evaluation_results"].get("rouge_1", 0)
                        for r in all_results.values()
                        if "rouge_1" in r["evaluation_results"]]
        if rouge_scores:
            avg_rouge = sum(rouge_scores) / len(rouge_scores)
            logger.info(f"Average ROUGE-1: {avg_rouge:.3f}")

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()