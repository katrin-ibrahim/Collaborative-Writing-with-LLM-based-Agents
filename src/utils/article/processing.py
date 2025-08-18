"""
Article quality validation and processing utilities.
"""

from pathlib import Path

import logging
from typing import Dict

from src.utils.data.models import Article

logger = logging.getLogger(__name__)


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


def validate_article_quality(content: str, min_words: int = 800) -> Dict:
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
