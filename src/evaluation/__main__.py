#!/usr/bin/env python3
"""
Main runner for evaluation module.
Evaluates articles from a results directory and adds evaluation metrics.
"""
import argparse
import concurrent.futures
import sys
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
    """Load existing results from results.json."""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        return json.load(f)


def save_results(results_dir: Path, data: Dict):
    """Save updated results back to results.json."""
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_article_content(results_dir: Path, method: str, topic: str) -> Optional[str]:
    """Get article content from file without storing in memory."""
    articles_dir = results_dir / "articles"

    # Create normalized topic name for file matching
    # Replace parentheses with underscores to match file naming convention
    normalized_topic = topic.replace("(", "").replace(")", "")

    # Try method_topic.md format with normalized topic
    article_file = articles_dir / f"{method}_{normalized_topic}.md"
    if article_file.exists():
        with open(article_file, "r") as f:
            return f.read()

    # Try method_topic.md format with original topic
    article_file = articles_dir / f"{method}_{topic}.md"
    if article_file.exists():
        with open(article_file, "r") as f:
            return f.read()

    # Try method/topic.md format with normalized topic
    article_file = articles_dir / method / f"{normalized_topic}.md"
    if article_file.exists():
        with open(article_file, "r") as f:
            return f.read()

    # Try method/topic.md format with original topic
    article_file = articles_dir / method / f"{topic}.md"
    if article_file.exists():
        with open(article_file, "r") as f:
            return f.read()

    return None


def evaluate_single_article(
    topic,
    method,
    entry,
    method_result_dict,
    results_dir_str,
    gen_entities=None,
    ref_entities=None,
):
    """Evaluate a single article - thread-safe function."""
    from src.evaluation.evaluator import ArticleEvaluator
    from src.utils.data_models import Article

    try:
        evaluator = ArticleEvaluator()
        evaluator.entity_metrics._precomputed_entities = {
            "generated": gen_entities,
            "reference": ref_entities,
        }

        # Get article content from file path
        if "article_path" in method_result_dict:
            article_path = Path(results_dir_str) / method_result_dict["article_path"]
            with open(article_path, "r") as f:
                article_content = f.read()
        elif "article" in method_result_dict:
            article_data = method_result_dict["article"]
            if isinstance(article_data, dict):
                article_content = article_data.get("content", "")
            else:
                article_content = str(article_data)
        else:
            # Fallback to get_article_content functionality
            results_dir = Path(results_dir_str)
            article_content = get_article_content(results_dir, method, topic)

        if not article_content:
            return (
                topic,
                method,
                None,
                f"Could not find article content for {method}/{topic}",
            )

        # Create Article object and evaluate
        article = Article(title=topic, content=article_content)
        eval_results = evaluator.evaluate_article(article, entry)

        return (topic, method, eval_results, None)

    except Exception as e:
        return (topic, method, None, str(e))


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("🔍 Article Evaluation Runner")

    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info(f"📂 Results directory: {results_dir}")

    # Load existing results or create new structure
    try:
        data = load_results(results_dir)
        logger.info("📄 Loaded existing results")
    except FileNotFoundError:
        logger.info("📝 No existing results.json found, creating new one")

        # Determine methods from directory structure
        articles_dir = results_dir / "articles"
        if not articles_dir.exists():
            logger.error(f"❌ Articles directory not found: {articles_dir}")
            return 1

        # Try to infer methods from files in the articles folder
        dir_name = results_dir.name
        methods = set()

        # First check for articles
        has_articles = False
        for item in articles_dir.iterdir():
            has_articles = True
            if item.is_file() and item.suffix == ".md":
                # Extract method from filename (e.g., direct_TopicName.md -> direct)
                method = item.stem.split("_")[0]
                methods.add(method)
            elif item.is_dir():
                # Also check for subdirectories as methods
                methods.add(item.name)

        # If no articles, check if methods can be inferred from directory name
        if not has_articles:
            # Format: all_N=1_T=19.07_22:39 or direct+storm_N=5_T=date_time
            dir_parts = dir_name.split("_")[0]
            if dir_parts != "all":
                # Extract methods from directory name (e.g., direct+storm -> [direct, storm])
                for method in dir_parts.split("+"):
                    methods.add(method)
            else:
                # Fallback to common methods
                methods.update(["direct", "storm", "rag"])
                logger.warning(
                    "⚠️ No articles found, using default methods: direct, storm, rag"
                )

        methods = list(methods)

        if not methods:
            logger.error("❌ Could not determine methods from directory structure")
            return 1

        logger.info(f"📊 Inferred methods from articles: {methods}")

        # Create new results structure
        # Extract timestamp from directory name if possible
        timestamp = "unknown"
        if "T=" in dir_name:
            timestamp = dir_name.split("T=")[1]

        from datetime import datetime

        data = {
            "summary": {
                "methods": {},  # Will be populated with model configs per method
                "timestamp": datetime.now().isoformat(),
                "total_time": 0.0,  # Will be calculated from individual generation times
            },
            "results": {},
        }

        # Scan articles directory to populate topics
        for method in methods:
            # First check for method_topic.md files
            for article_file in articles_dir.glob(f"{method}_*.md"):
                # Extract topic from filename
                filename = article_file.stem
                # Remove method prefix
                topic_part = filename[len(method) + 1 :]

                # Check if this is a topic with slashes that were replaced with underscores
                if "and_or" in topic_part:
                    topic = topic_part.replace("and_or", "and/or")
                else:
                    topic = topic_part

                if topic not in data["results"]:
                    data["results"][topic] = {}

                # Try to load metadata file
                base_filename = article_file.stem
                metadata_file = articles_dir / f"{base_filename}_metadata.json"

                generation_time = 10.0  # Dummy value for backward compatibility
                word_count = 0
                model_info = "unknown"

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            generation_time = metadata.get("generation_time", 10.0)
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

                data["results"][topic][method] = {
                    "success": True,
                    "generation_time": generation_time,
                    "word_count": word_count,
                    "article_path": str(article_file.relative_to(results_dir)),
                }

                # Track model config per method for summary
                if method not in data["summary"]["methods"]:
                    data["summary"]["methods"][method] = {
                        "model": model_info,
                        "article_count": 0,
                    }
                data["summary"]["methods"][method]["article_count"] += 1

            # Also check for method/topic.md structure
            method_dir = articles_dir / method
            if method_dir.exists() and method_dir.is_dir():
                for article_file in method_dir.glob("*.md"):
                    topic = article_file.stem
                    if topic not in data["results"]:
                        data["results"][topic] = {}

                    # Add to results structure without storing article content
                    data["results"][topic][method] = {
                        "success": True,
                        "article_path": str(article_file.relative_to(results_dir)),
                    }

        logger.info(
            f"📊 Discovered {len(data['results'])} topics across {len(methods)} methods"
        )

    # Calculate total generation time from individual article times
    total_generation_time = 0.0
    for topic_data in data["results"].values():
        for method_data in topic_data.values():
            total_generation_time += method_data.get("generation_time", 0.0)

    data["summary"]["total_time"] = total_generation_time

    logger.info(
        f"📊 Discovered {len(data['results'])} topics across {len(methods)} methods"
    )
    logger.info(f"🕒 Total generation time: {total_generation_time:.2f}s")

    # Load FreshWiki dataset for evaluation
    logger.info("📚 Loading FreshWiki dataset...")
    freshwiki = FreshWikiLoader()
    entries = freshwiki.get_evaluation_sample(1000)  # Load all entries

    # Create evaluator
    ArticleEvaluator()

    # Process each topic
    results = data.get("results", {})
    methods = data.get("configuration", {}).get("methods", [])

    skipped_count = 0
    evaluated_count = 0

    # Prepare evaluation tasks
    eval_tasks = []

    for topic, topic_results in results.items():
        # Find the corresponding FreshWiki entry
        entry = None
        for e in entries:
            # Try exact match first
            if e.topic == topic:
                entry = e
                break
            # Try handling cases where slashes were replaced with underscores
            normalized_topic = topic.lower().replace("_", " ")
            normalized_e_topic = e.topic.lower().replace("_", " ").replace("/", " ")
            if normalized_topic == normalized_e_topic:
                entry = e
                break
            if "and/or" in e.topic and topic.replace("and_or", "and/or") == e.topic:
                entry = e
                break

        if not entry:
            logger.warning(f"⚠️ No FreshWiki entry found for topic: {topic}")
            continue

        for method in methods:
            if method not in results[topic]:
                continue

            method_result = results[topic][method]
            if not method_result.get("success"):
                continue

            if "evaluation" in method_result and not args.force:
                skipped_count += 1
                continue

            # Prepare task for threading
            eval_tasks.append((topic, method, entry, method_result, str(results_dir)))

    logger.info(f"🧪 Preparing to evaluate {len(eval_tasks)} articles")

    if not eval_tasks:
        logger.info("✅ No articles to evaluate")
        save_results(results_dir, data)
        return 0

    # Pre-extract all entities in main thread to avoid threading issues with FLAIR
    logger.info("🧬 Pre-extracting entities for thread-safe evaluation...")
    from src.evaluation.metrics.entity_metrics import EntityMetrics

    entity_extractor = EntityMetrics()

    entity_cache = {}
    for topic, method, entry, method_result, results_dir_str in eval_tasks:
        if topic not in entity_cache:
            # Extract entities once per topic
            gen_entities = entity_extractor.extract_entities(
                get_article_content(Path(results_dir_str), method, topic) or ""
            )
            ref_entities = entity_extractor.extract_entities(entry.reference_content)
            entity_cache[topic] = (gen_entities, ref_entities)

    # Use ThreadPoolExecutor with pre-extracted entities
    max_workers = min(4, len(eval_tasks))  # Can now use more workers safely

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with pre-extracted entities
        future_to_task = {
            executor.submit(
                evaluate_single_article,
                topic,
                method,
                entry,
                method_result,
                results_dir_str,
                *entity_cache[topic],
            ): (topic, method, entry, method_result, results_dir_str)
            for topic, method, entry, method_result, results_dir_str in eval_tasks
        }

        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                topic, method, eval_results, error = future.result()
                method_result = results[topic][method]

                if eval_results is not None:
                    method_result["evaluation"] = eval_results
                    if "word_count" not in method_result and article_content:
                        method_result["word_count"] = len(article_content.split())
                    logger.info(f"✅ Evaluated {method} for {topic}")
                    evaluated_count += 1
                else:
                    logger.error(f"❌ Evaluation failed for {method}/{topic}: {error}")
                    method_result["evaluation_error"] = error

            except Exception as e:
                task = future_to_task[future]
                topic, method = task[0], task[1]
                logger.error(f"❌ Task execution failed for {method}/{topic}: {e}")

    # Save updated results
    save_results(results_dir, data)

    # Summary
    logger.info(f"📊 Evaluation Summary:")
    logger.info(f"  - Evaluated: {evaluated_count} articles")
    logger.info(f"  - Skipped: {skipped_count} articles")
    logger.info(f"💾 Updated results saved to: {results_dir / 'results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
