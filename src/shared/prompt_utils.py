"""
Shared prompt utilities for article generation.
Contains utilities needed by both local and baseline implementations.
"""

import sys
import time
from pathlib import Path

import re

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.data_models import Article


def build_direct_prompt(topic: str) -> str:
    """Build a concise direct prompting template for fast article generation."""
    return f"""Write a comprehensive Wikipedia-style article about "{topic}".

Requirements:
1. Start with a clear introduction summarizing key facts
2. Create 4-5 specific section headings (not generic ones like "Overview")
3. Include specific details, dates, names, and facts
4. Use encyclopedic tone
5. Target 800-1200 words
6. Format as markdown with # {topic} as title

Write the article now:"""


def post_process_article(content: str, topic: str) -> str:
    """Post-process article content to improve quality and structure."""
    if not content or not content.strip():
        return content

    lines = content.split("\n")
    processed_lines = []

    # Ensure proper title formatting
    if not lines[0].strip().startswith("#"):
        processed_lines.append(f"# {topic}")
        processed_lines.append("")

    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")
            continue

        # Fix heading hierarchy
        if line.startswith("#"):
            # Ensure proper spacing after headings
            if line.startswith("##") and not line.startswith("### "):
                processed_lines.append(line)
                processed_lines.append("")
            else:
                processed_lines.append(line)
                processed_lines.append("")
        else:
            processed_lines.append(line)

    # Join and clean up excessive whitespace
    processed_content = "\n".join(processed_lines)
    # Remove excessive blank lines
    processed_content = re.sub(r"\n\n\n+", "\n\n", processed_content)

    return processed_content.strip()


def error_article(topic: str, error: str, method: str) -> Article:
    """Create an error article using the shared Article model."""
    return Article(
        title=topic,
        content=f"# {topic}\n\nError: {error}",
        sections={},
        metadata={
            "method": method,
            "error": error,
            "generation_time": 0.0,
            "timestamp": time.time(),
            "word_count": 0,
        },
    )


def count_words(text: str) -> int:
    """Count words in text content."""
    if not text:
        return 0
    # Remove markdown headers and clean text
    cleaned_text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    words = cleaned_text.split()
    return len(words)
