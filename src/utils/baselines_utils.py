# FILE: utils/baseline_utils.py
"""
Shared utility functions for baseline experiments.
Used by both Ollama and local baseline implementations to maintain DRY principles.
"""
from pathlib import Path

import logging
from typing import Dict, List

from src.utils.data_models import Article

logger = logging.getLogger(__name__)


def build_direct_prompt(topic: str) -> str:
    """Build direct prompting prompt (shared between Ollama and local baselines)."""
    return f"""Write a comprehensive, well-structured Wikipedia-style article about \"{topic}\".

You are an expert encyclopedia writer. Create a detailed, factual article that captures the essential information about this topic.

CRITICAL REQUIREMENTS:
1. Start with a strong, informative introduction that summarizes the key facts
2. Create 4-6 main sections with specific, descriptive headings (NOT generic ones like "Overview")
3. Each section should contain 2-3 substantial paragraphs with specific details
4. Include dates, numbers, names, and concrete facts wherever possible
5. Use proper Wikipedia-style citations format [1], [2], etc. (even if hypothetical)
6. Maintain an encyclopedic, neutral tone throughout
7. Target 1200-1600 words for comprehensive coverage
8. Include entity-rich content with proper nouns, technical terms, and specific details

SECTION STRATEGY:
- Choose section headings that are specific to the topic domain
- For events: Background, Timeline, Key figures, Impact, Aftermath
- For organizations: History, Structure, Operations, Services, Controversies
- For people: Early life, Career, Major achievements, Legacy
- For places: Geography, History, Demographics, Economy, Culture
- For concepts: Definition, Development, Applications, Criticism

FORMAT:
# {topic}

[Write a comprehensive 2-3 paragraph introduction that defines the topic, explains its significance, and provides key contextual information. Include specific dates, locations, and quantitative details.]

## [Section 1 - Specific heading related to topic]

[2-3 detailed paragraphs with specific facts, dates, names, and quantitative information. Include proper citations.]

## [Section 2 - Another specific heading]

[2-3 detailed paragraphs continuing the comprehensive coverage.]

## [Section 3 - Third specific heading]

[2-3 detailed paragraphs with continued depth and specificity.]

## [Section 4 - Fourth specific heading]

[2-3 detailed paragraphs maintaining encyclopedic quality.]

## [Section 5 - Fifth specific heading if needed]

[2-3 detailed paragraphs for comprehensive coverage.]

## [Section 6 - Final specific heading if needed]

[2-3 detailed paragraphs completing the comprehensive article.]

Write the complete article now."""


def error_article(topic: str, error_msg: str, method: str) -> Article:
    """Create error article when generation fails (shared utility)."""
    return Article(
        title=topic,
        content=f"# {topic}\n\nError generating article: {error_msg}",
        sections={},
        metadata={
            "method": method,
            "error": True,
            "error_message": error_msg,
            "word_count": 0,
            "generation_time": 0.0,
        },
    )


def extract_storm_output(storm_output_dir: Path, topic: str) -> str:
    """Extract STORM polished article content only."""
    try:
        # STORM creates the polished article here
        polished_file = (
            storm_output_dir
            / topic.replace(" ", "_").replace("/", "_")
            / "storm_gen_article_polished.txt"
        )

        if polished_file.exists():
            with open(polished_file, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Polished article not found: {polished_file}")

    except Exception as e:
        logger.error(f"Failed to extract STORM output: {e}")
        return f"# {topic}\n\nError reading STORM output: {e}"


def build_rag_prompt(topic: str, context: str) -> str:
    """Build RAG prompt with retrieved context (shared utility)."""
    return f"""Write a comprehensive Wikipedia-style article about "{topic}" using the provided context.

Context Information:
{context}

Guidelines:
- Use the context to write an accurate, well-structured article
- Organize information into clear sections
- Write in encyclopedic style
- Focus on factual, verifiable information
- Create a comprehensive overview of the topic

Write the article:"""


def enhance_content_prompt(topic: str, content: str) -> str:
    """Build enhancement prompt for short content (shared utility)."""
    return f"""The following article about "{topic}" needs to be enhanced and expanded to meet Wikipedia standards.

Current article:
{content}

Please rewrite and expand this article to be comprehensive, well-structured, and informative. Focus on:
- Adding missing important information
- Improving organization and flow
- Ensuring factual accuracy
- Using encyclopedic tone
- Creating proper sections and subsections

Write the enhanced article:"""


def validate_article_quality(content: str, min_words: int = 800) -> dict:
    """Validate article quality and return metrics (shared utility)."""
    if not content:
        return {"valid": False, "reason": "Empty content", "word_count": 0}

    word_count = len(content.split())

    # Check minimum word count
    if word_count < min_words:
        return {
            "valid": False,
            "reason": f"Too short ({word_count} words, minimum {min_words})",
            "word_count": word_count,
        }

    # Check for proper heading structure
    lines = content.split("\n")
    has_main_heading = any(line.strip().startswith("# ") for line in lines)
    has_sections = any(line.strip().startswith("## ") for line in lines)

    if not has_main_heading:
        return {
            "valid": False,
            "reason": "Missing main heading",
            "word_count": word_count,
        }

    if not has_sections:
        return {
            "valid": False,
            "reason": "Missing section headings",
            "word_count": word_count,
        }

    return {
        "valid": True,
        "word_count": word_count,
        "has_main_heading": has_main_heading,
        "has_sections": has_sections,
    }


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
    from src.utils.output_manager import OutputManager

    if args.resume_dir:
        # Resume from specific directory
        resume_dir = OutputManager.verify_resume_dir(args.resume_dir)
        logger.info(f"📂 Resuming from specified directory: {resume_dir}")
        return Path(resume_dir)

    # Check if custom output directory is specified
    if hasattr(args, "output_dir") and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Using custom output directory: {output_dir}")
        return output_dir

    # If no custom output_dir specified, create new experiment directory using OutputManager
    custom_name = (
        getattr(args, "experiment_name", None)
        if hasattr(args, "experiment_name")
        else None
    )
    output_path = OutputManager.create_output_dir(
        args.backend, args.methods, args.num_topics, custom_name=custom_name
    )
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if custom_name:
        logger.info(f"📂 Created new run directory with custom name: {output_dir}")
    else:
        logger.info(f"📂 Created new run directory: {output_dir}")
    return output_dir


def make_serializable(obj):
    if hasattr(obj, "__dict__"):
        return {k: make_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj


def save_final_results(
    output_dir: Path,
    topics: List[str],
    methods: List[str],
    total_time: float,
    backend: str = "unknown",
) -> bool:
    """
    Generate and save final results.json file from completed experiment.

    Scans the articles directory to create a complete results structure.
    """
    from datetime import datetime

    import json

    results_file = output_dir / "results.json"
    articles_dir = output_dir / "articles"

    logger.info(f"Saving final results to: {results_file}")

    # Build results structure
    data = {
        "summary": {
            "timestamp": datetime.now().isoformat(),
            "backend": backend,
            "methods": {},
            "total_time": total_time,
            "topics_processed": len(topics),
        },
        "results": {},
    }

    # Initialize all topics
    for topic in topics:
        data["results"][topic] = {}

    # Scan articles directory for actual results
    if articles_dir.exists():
        for method in methods:
            method_articles = 0

            # Check for method_topic.md pattern
            for article_file in articles_dir.glob(f"{method}_*.md"):
                topic_part = article_file.stem[len(method) + 1 :]

                # Handle topics with slashes that were replaced with underscores
                if "and_or" in topic_part:
                    topic = topic_part.replace("and_or", "and/or")
                else:
                    topic = topic_part.replace("_", " ")

                # Load metadata if available
                metadata_file = articles_dir / f"{article_file.stem}_metadata.json"
                generation_time = 0.0
                word_count = 0
                model_info = "unknown"

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            generation_time = metadata.get("generation_time", 0.0)
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

                # Ensure topic exists in results
                if topic not in data["results"]:
                    data["results"][topic] = {}

                data["results"][topic][method] = {
                    "success": True,
                    "generation_time": generation_time,
                    "word_count": word_count,
                    "article_path": str(article_file.relative_to(output_dir)),
                }
                method_articles += 1

            # Check for method/topic.md structure
            method_dir = articles_dir / method
            if method_dir.exists() and method_dir.is_dir():
                for article_file in method_dir.glob("*.md"):
                    topic = article_file.stem.replace("_", " ")

                    if topic not in data["results"]:
                        data["results"][topic] = {}

                    data["results"][topic][method] = {
                        "success": True,
                        "article_path": str(article_file.relative_to(output_dir)),
                    }
                    method_articles += 1

            # Update method summary
            data["summary"]["methods"][method] = {
                "model": model_info,
                "article_count": method_articles,
            }

    # Mark missing results as failed
    for topic in topics:
        for method in methods:
            if method not in data["results"][topic]:
                data["results"][topic][method] = {
                    "success": False,
                    "error": f"No {method} result found for topic",
                }

    try:
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"✅ Final results saved to: {results_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")
        return False
