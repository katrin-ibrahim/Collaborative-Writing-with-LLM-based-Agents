"""
Article quality validation and processing utilities.
"""

from pathlib import Path

import logging

logger = logging.getLogger(__name__)


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
