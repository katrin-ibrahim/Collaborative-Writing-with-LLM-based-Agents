#!/usr/bin/env python3
"""
Main runner for evaluation module.
Evaluates articles from a results directory and adds evaluation metrics.
"""
import argparse
import sys
import time
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


def main():
    """Main evaluation function."""
    start_time = time.time()

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

        # Extract methods from existing data structure
        if "summary" in data and "methods" in data["summary"]:
            methods = list(data["summary"]["methods"].keys())
        elif "configuration" in data and "methods" in data["configuration"]:
            methods = data["configuration"]["methods"]
        else:
            # Infer methods from results structure
            methods = set()
            for topic_data in data.get("results", {}).values():
                methods.update(topic_data.keys())
            methods = list(methods)

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
    entries = freshwiki.load_topics(100)  # Load all entries

    # Create evaluator
    evaluator = ArticleEvaluator()

    # Process each topic
    results = data.get("results", {})

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

            # Try handling cases where slashes were replaced with underscores
            normalized_topic = topic.lower().replace("_", " ")
            normalized_e_topic = e.topic.lower().replace("_", " ").replace("/", " ")
            if normalized_topic == normalized_e_topic:
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

            # Prepare task for evaluation
            eval_tasks.append((topic, method, entry, method_result, str(results_dir)))

    logger.info(f"🧪 Preparing to evaluate {len(eval_tasks)} articles")

    if not eval_tasks:
        logger.info("✅ No articles to evaluate")
        save_results(results_dir, data)
        return 0

    # Evaluate articles sequentially (no threading)
    # Evaluate articles sequentially (no threading)
    logger.info(f"🧪 Evaluating {len(eval_tasks)} articles sequentially...")

    for topic, method, entry, method_result, results_dir_str in eval_tasks:
        try:
            article_content = None
            if "article_path" in method_result:
                article_path = Path(results_dir_str) / method_result["article_path"]
                try:
                    with open(article_path, "r") as f:
                        article_content = f.read()
                        "/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/results/ollama/storm_N=1_T=19.07_23:11/articles/storm_Music_written_in_all_major_and_or_minor_keys.md"
                        "/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/results/ollama/storm_N=1_T=19.07_23:11/articles/storm_Music_written_in_all_major_and_or_minor_keys.md"
                    logger.debug(f"✅ Read article from: {article_path}")
                except Exception as e:
                    logger.error(f"❌ Could not read file {article_path}: {e}")
                    article_content = None
            else:
                # Fallback to old method if no article_path stored
                logger.warning(
                    f"⚠️ No article_path found for {method}/{topic}, using fallback"
                )
                article_content = get_article_content(
                    Path(results_dir_str), method, topic
                )

            if not article_content:
                logger.error(f"❌ Could not find article content for {method}/{topic}")
                # Provide zero evaluation results for data_loader compatibility
                method_result["evaluation"] = {
                    "rouge_1": 0.0,
                    "rouge_l": 0.0,
                    "heading_soft_recall": 0.0,
                    "heading_entity_recall": 0.0,
                    "article_entity_recall": 0.0,
                }
                method_result["evaluation_error"] = "Could not find article content"
                continue

            # Create Article object and evaluate
            from src.utils.data_models import Article

            article = Article(title=topic, content=article_content)

            # FIX 2 & 3: Add debug logging to evaluator for HER and HSR debugging
            logger.debug(f"🔍 Starting evaluation for {method}/{topic}")
            logger.debug(f"📄 Article content length: {len(article_content)} chars")

            eval_results = evaluator.evaluate_article(article, entry)

            if eval_results is not None:
                method_result["evaluation"] = eval_results
                # Ensure word_count is present for data_loader compatibility
                if "word_count" not in method_result:
                    method_result["word_count"] = len(article_content.split())

                # FIX 2: Debug logging for Heading Entity Recall
                logger.debug(f"🎯 Evaluation results for {method}/{topic}:")
                logger.debug(f"   📊 ROUGE-1: {eval_results.get('rouge_1', 0):.2f}%")
                logger.debug(
                    f"   📊 Heading Soft Recall: {eval_results.get('heading_soft_recall', 0):.2f}%"
                )
                logger.debug(
                    f"   📊 Heading Entity Recall: {eval_results.get('heading_entity_recall', 0):.2f}%"
                )
                logger.debug(
                    f"   📊 Article Entity Recall: {eval_results.get('article_entity_recall', 0):.2f}%"
                )

                # Add manual debugging for HER if it's 0
                if eval_results.get("heading_entity_recall", 0) == 0:
                    logger.warning(
                        f"🔍 DEBUG: HER is 0 for {method}/{topic} - investigating..."
                    )
                    # Extract headings manually for debugging
                    from src.evaluation.metrics.heading_metrics import HeadingMetrics

                    heading_extractor = HeadingMetrics()
                    generated_headings = (
                        heading_extractor.extract_headings_from_content(article_content)
                    )
                    logger.debug(f"   🏷️ Generated headings: {generated_headings}")
                    logger.debug(f"   🏷️ Reference headings: {entry.reference_outline}")

                    # Test entity extraction on headings
                    from src.evaluation.metrics.entity_metrics import EntityMetrics

                    entity_extractor = EntityMetrics()

                    # Extract entities from each generated heading
                    gen_heading_entities = set()
                    for heading in generated_headings:
                        heading_entities = entity_extractor.extract_entities(heading)
                        gen_heading_entities.update(heading_entities)
                        if heading_entities:
                            logger.debug(
                                f"   🔍 Heading '{heading}' entities: {heading_entities}"
                            )
                        else:
                            logger.debug(f"   🔍 Heading '{heading}' entities: NONE")

                    # Extract entities from each reference heading
                    ref_heading_entities = set()
                    for heading in entry.reference_outline:
                        heading_entities = entity_extractor.extract_entities(heading)
                        ref_heading_entities.update(heading_entities)
                        if heading_entities:
                            logger.debug(
                                f"   🔍 Reference heading '{heading}' entities: {heading_entities}"
                            )
                        else:
                            logger.debug(
                                f"   🔍 Reference heading '{heading}' entities: NONE"
                            )

                    logger.debug(
                        f"   🔍 Total generated heading entities: {gen_heading_entities}"
                    )
                    logger.debug(
                        f"   🔍 Total reference heading entities: {ref_heading_entities}"
                    )
                    logger.debug(
                        f"   🔍 Overlap: {gen_heading_entities.intersection(ref_heading_entities)}"
                    )

                # Add manual debugging for HSR if it seems too high
                hsr_value = eval_results.get("heading_soft_recall", 0)
                if hsr_value > 90:
                    logger.warning(
                        f"🔍 DEBUG: HSR is very high ({hsr_value:.2f}%) for {method}/{topic} - investigating..."
                    )
                    # Get the actual soft cardinality values for manual verification
                    from src.evaluation.metrics.heading_metrics import HeadingMetrics

                    heading_metrics = HeadingMetrics()
                    generated_headings = heading_metrics.extract_headings_from_content(
                        article_content
                    )

                    # Calculate HSR manually with debug logging
                    logger.debug(f"   🔢 Manual HSR calculation:")
                    logger.debug(
                        f"   📝 Generated headings ({len(generated_headings)}): {generated_headings}"
                    )
                    logger.debug(
                        f"   📝 Reference headings ({len(entry.reference_outline)}): {entry.reference_outline}"
                    )

                    # Get embeddings and calculate cardinalities manually
                    if generated_headings and entry.reference_outline:
                        import numpy as np

                        ref_embeddings = heading_metrics.embedder.encode(
                            entry.reference_outline
                        )
                        gen_embeddings = heading_metrics.embedder.encode(
                            generated_headings
                        )
                        all_embeddings = np.vstack([ref_embeddings, gen_embeddings])

                        ref_card = heading_metrics._calculate_soft_cardinality(
                            ref_embeddings
                        )
                        gen_card = heading_metrics._calculate_soft_cardinality(
                            gen_embeddings
                        )
                        union_card = heading_metrics._calculate_soft_cardinality(
                            all_embeddings
                        )
                        intersection_card = ref_card + gen_card - union_card
                        manual_hsr = (
                            intersection_card / ref_card if ref_card > 0 else 0.0
                        )

                        logger.debug(
                            f"   🔢 Manual calculation: ref_card={ref_card:.6f}, gen_card={gen_card:.6f}"
                        )
                        logger.debug(
                            f"   🔢 Manual calculation: union_card={union_card:.6f}, intersection_card={intersection_card:.6f}"
                        )
                        logger.debug(
                            f"   🔢 Manual HSR: {manual_hsr:.6f} ({manual_hsr*100:.2f}%)"
                        )
                        logger.debug(f"   🔢 Reported HSR: {hsr_value:.2f}%")

                logger.info(f"✅ Evaluated {method} for {topic}")
                evaluated_count += 1
            else:
                logger.error(
                    f"❌ Evaluation failed for {method}/{topic}: evaluation returned None"
                )
                # Provide zero evaluation results for data_loader compatibility
                method_result["evaluation"] = {
                    "rouge_1": 0.0,
                    "rouge_l": 0.0,
                    "heading_soft_recall": 0.0,
                    "heading_entity_recall": 0.0,
                    "article_entity_recall": 0.0,
                }
                method_result["evaluation_error"] = "Evaluation returned None"

        except Exception as e:
            logger.error(f"❌ Evaluation failed for {method}/{topic}: {e}")
            # Provide zero evaluation results for data_loader compatibility
            method_result["evaluation"] = {
                "rouge_1": 0.0,
                "rouge_l": 0.0,
                "heading_soft_recall": 0.0,
                "heading_entity_recall": 0.0,
                "article_entity_recall": 0.0,
            }
            method_result["evaluation_error"] = str(e)

    # Calculate total evaluation time
    total_time = time.time() - start_time

    # Add evaluation summary to data if summary section exists
    if "summary" in data:
        data["summary"].update(
            {
                "evaluation_time": total_time,
                "topics_evaluated": evaluated_count,
                "topics_skipped": skipped_count,
                "total_topics": len(
                    [
                        t
                        for t in results.keys()
                        if any(method in results[t] for method in methods)
                    ]
                ),
            }
        )

    # Save updated results
    save_results(results_dir, data)

    # Summary
    logger.info(f"📊 Evaluation Summary:")
    logger.info(f"  - Total evaluation time: {total_time:.2f}s")
    logger.info(f"  - Evaluated: {evaluated_count} articles")
    logger.info(f"  - Skipped: {skipped_count} articles")
    logger.info(f"💾 Updated results saved to: {results_dir / 'results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
