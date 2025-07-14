#!/usr/bin/env python3
"""
Script to regenerate evaluation results from article files.
Reads all direct_*.md and storm_*.md files and creates a complete results.json
with evaluation metrics but without article content.
"""

import sys
from datetime import datetime
from pathlib import Path

import json
import os
import re
from typing import Any, Dict

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))


def count_words(text: str) -> int:
    """Count words in text, excluding markdown formatting."""
    # Remove markdown headers, links, etc.
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`]", "", text)

    # Count words
    words = text.split()
    return len(words)


def extract_topic_from_filename(filename: str) -> str:
    """Extract topic name from filename (remove direct_/storm_ prefix and .md suffix)."""
    if filename.startswith("direct_"):
        return filename[7:-3]  # Remove 'direct_' and '.md'
    elif filename.startswith("storm_"):
        return filename[6:-3]  # Remove 'storm_' and '.md'
    else:
        return filename[:-3]  # Just remove '.md'


def read_article_file(filepath: Path) -> Dict[str, Any]:
    """Read article file and extract basic info."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract title (first line should be # Title)
        lines = content.split("\n")
        title = ""
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()

        word_count = count_words(content)

        return {
            "success": True,
            "word_count": word_count,
            "article": {
                "title": title,
                "metadata": {
                    "word_count": word_count,
                    "method": (
                        "direct" if filepath.name.startswith("direct_") else "storm"
                    ),
                },
            },
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to read article: {str(e)}"}


def load_reference_data(topic_name):
    """Load reference data for a topic directly from FreshWiki JSON and TXT files."""
    json_path = f"/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/data/freshwiki/json/{topic_name}.json"
    txt_path = f"/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/data/freshwiki/txt/{topic_name}.txt"

    if not os.path.exists(json_path) or not os.path.exists(txt_path):
        print(f"    Reference data not found for topic: {topic_name}")
        return None

    try:
        # Load JSON metadata
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Load text content
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create FreshWikiEntry object
        from utils.freshwiki_loader import FreshWikiEntry

        return FreshWikiEntry(
            topic=json_data.get("title", topic_name),
            reference_outline=json_data.get("sections", []),
            reference_content=content,
            metadata={
                "url": json_data.get("url", ""),
                "summary": json_data.get("summary", ""),
                "word_count": json_data.get("word_count", len(content.split())),
                "source_file": json_data.get("source_file", ""),
                "quality_filtered": True,
            },
        )
    except Exception as e:
        print(f"    Error loading reference data for {topic_name}: {e}")
        return None


def run_evaluation_on_article(article_content: str, topic: str) -> Dict[str, float]:
    """
    Run evaluation metrics on an article using the existing evaluator and FreshWiki reference data.
    """
    try:
        # Import your evaluation modules
        from evaluation.evaluator import ArticleEvaluator
        from utils.data_models import Article

        print(f"    Running evaluation for {topic}...")

        # Load reference data for the topic
        reference_data = load_reference_data(topic)

        if not reference_data:
            print(f"    Warning: No reference data found for topic {topic}")
            return {
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0,
                "heading_soft_recall": 0.0,
                "heading_entity_recall": 0.0,
                "article_entity_recall": 0.0,
            }

        # Create article data structure that the evaluator expects
        article = Article(title=topic.replace("_", " "), content=article_content)

        # Create evaluator and run evaluation
        evaluator = ArticleEvaluator()
        evaluation_results = evaluator.evaluate_article(
            article=article, reference=reference_data
        )

        print(f"    ✓ Full evaluation completed for {topic}")
        return evaluation_results

    except ImportError as e:
        print(f"    Warning: Failed to import evaluation modules for {topic}: {str(e)}")
        return {}
    except Exception as e:
        print(f"    Warning: Evaluation failed for topic {topic}: {str(e)}")
        return {}


def main():
    """Main function to regenerate evaluation results."""
    articles_dir = Path("results/ollama/run_20250709_104159/articles")
    output_file = Path("regenerated_results.json")

    print("Regenerating evaluation results from article files...")
    print(f"Articles directory: {articles_dir.absolute()}")
    print(f"Output file: {output_file.absolute()}")

    if not articles_dir.exists():
        print(f"Error: Articles directory not found: {articles_dir}")
        return

    # Find all article files
    direct_files = list(articles_dir.glob("direct_*.md"))
    storm_files = list(articles_dir.glob("storm_*.md"))

    print(
        f"Found {len(direct_files)} direct articles and {len(storm_files)} storm articles"
    )

    # Create results structure
    results = {}
    successful_evaluations = 0
    total_articles = 0

    # Process all topics
    all_topics = set()
    for f in direct_files:
        all_topics.add(extract_topic_from_filename(f.name))
    for f in storm_files:
        all_topics.add(extract_topic_from_filename(f.name))

    print(f"Processing {len(all_topics)} unique topics...")

    for i, topic in enumerate(sorted(all_topics), 1):
        print(f"Processing topic {i}/{len(all_topics)}: {topic}")
        results[topic] = {}

        # Process direct method
        direct_file = articles_dir / f"direct_{topic}.md"
        if direct_file.exists():
            total_articles += 1
            article_data = read_article_file(direct_file)

            if article_data["success"]:
                # Read article content for evaluation
                with open(direct_file, "r", encoding="utf-8") as f:
                    article_content = f.read()

                # Run evaluation (without reference for now - could compare direct vs storm later)
                evaluation_metrics = run_evaluation_on_article(article_content, topic)

                results[topic]["direct"] = {
                    "success": True,
                    "word_count": article_data["word_count"],
                    "metrics": evaluation_metrics,
                }

                if evaluation_metrics:
                    successful_evaluations += 1
            else:
                results[topic]["direct"] = article_data

        # Process storm method
        storm_file = articles_dir / f"storm_{topic}.md"
        if storm_file.exists():
            total_articles += 1
            article_data = read_article_file(storm_file)

            if article_data["success"]:
                # Read article content for evaluation
                with open(storm_file, "r", encoding="utf-8") as f:
                    article_content = f.read()

                # Run evaluation (without reference for now - could compare direct vs storm later)
                evaluation_metrics = run_evaluation_on_article(article_content, topic)

                results[topic]["storm"] = {
                    "success": True,
                    "word_count": article_data["word_count"],
                    "metrics": evaluation_metrics,
                }

                if evaluation_metrics:
                    successful_evaluations += 1
            else:
                results[topic]["storm"] = article_data

    # Create complete results JSON structure
    complete_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "methods": ["direct", "storm"],
            "num_topics": len(all_topics),
            "regenerated_from": "article_files",
            "source_directory": str(articles_dir.absolute()),
        },
        "results": results,
        "summary": {
            "total_topics": len(all_topics),
            "total_articles": total_articles,
            "successful_evaluations": successful_evaluations,
            "methods_run": ["direct", "storm"],
        },
    }

    # Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print("REGENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total topics: {len(all_topics)}")
    print(f"Total articles processed: {total_articles}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Results saved to: {output_file.absolute()}")

    if successful_evaluations < total_articles:
        print(
            f"\nNote: {total_articles - successful_evaluations} articles failed evaluation"
        )
        print(
            "This might be due to missing reference data or evaluation pipeline issues"
        )


if __name__ == "__main__":
    main()
