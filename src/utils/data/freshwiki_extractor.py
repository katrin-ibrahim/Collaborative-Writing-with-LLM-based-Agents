# src/utils/extract_quality_freshwiki.py
import random
from pathlib import Path

import json
import logging
import os
import re

logger = logging.getLogger(__name__)


def is_quality_topic(title: str, content: str) -> bool:
    """Check if a topic meets quality standards."""
    if not title or not content:
        return False

    title = title.strip()
    content = content.strip()

    # Basic length requirements
    if len(title) < 3 or len(title) > 200:
        return False

    word_count = len(content.split())
    if word_count < 100:  # At least 100 words
        return False

    # Filter out obvious non-topics
    invalid_patterns = [
        r"^https?://",  # URLs
        r"^\[[\d]+\]",  # Reference numbers like [1]
        r"^#+\s*$",  # Just markdown headers
        r"\.com",  # Domain names
        r"^(javascript|css|html):",  # Code snippets
        r"^(In the text|are preceded)",  # Text fragments
        r"^#\s*(Aftermath|Crash|Victims)$",  # Just single word headers
    ]

    for pattern in invalid_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return False

    # Must not be just numbers or special characters
    if re.match(r"^[\d\s\-_\.]+$", title):
        return False

    # Content quality checks
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    # Should have some structure (headings)
    headings = [line for line in lines if line.startswith("#")]
    if len(headings) < 2:
        return False

    # Should have substantial text content (not just headings)
    text_lines = [
        line for line in lines if line and not line.startswith("#") and len(line) > 10
    ]
    if len(text_lines) < 5:
        return False

    # Check for real content vs just metadata
    content_paragraphs = [line for line in text_lines if len(line.split()) > 5]
    if len(content_paragraphs) < 3:
        return False

    return True


def extract_sections(content: str) -> list:
    """Extract clean section titles from content."""
    sections = []
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            if section and len(section) > 1 and section not in sections:
                sections.append(section)

    return sections


def copy_file_content(content: str, dst_path: Path):
    """Copy content to destination, ensuring it's a real file."""
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to copy to {dst_path}: {e}")
        return False


def find_freshwiki_files(source_dir: Path):
    print(source_dir)
    """Find all potential FreshWiki content files."""
    files = []

    # Look for common patterns in FreshWiki repos
    patterns = [
        "**/*.txt",
        "**/*.md",
        "**/*.json",
        "**/data/**/*",
        "**/articles/**/*",
        "**/texts/**/*",
    ]

    for pattern in patterns:
        for file_path in source_dir.glob(pattern):
            if (
                file_path.is_file() and file_path.stat().st_size > 100
            ):  # At least 100 bytes
                files.append(file_path)

    logger.info(f"Found {len(files)} potential content files")
    return files


def extract_quality_topics(source_dir: str, target_dir: str, max_topics: int = 30):
    """Extract quality topics from cloned FreshWiki repo."""

    logger.info(
        "Starting quality topic extraction, the current working directory is: %s",
        os.getcwd(),
    )

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        return False

    logger.info(f"Extracting quality topics from: {source_path}")
    logger.info(f"Target directory: {target_path}")

    # Create target directories
    json_dir = target_path / "json"
    txt_dir = target_path / "txt"
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Find all potential content files
    content_files = find_freshwiki_files(source_path)

    if not content_files:
        logger.error("No content files found. Check the source directory structure.")
        return False

    # Process files and find quality topics
    quality_topics = []
    processed = 0

    for file_path in content_files:
        if len(quality_topics) >= max_topics * 3:  # Get more candidates than needed
            break

        try:
            processed += 1

            # Read file content immediately (while it's not a symlink)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                continue

            # Try to extract title
            lines = content.split("\n")
            title = ""

            # Try first non-empty line
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip().lstrip("#").strip()
                if line and len(line) > 3:
                    title = line
                    break

            # Fallback to filename
            if not title:
                title = file_path.stem.replace("_", " ").replace("-", " ")

            # Quality check
            if is_quality_topic(title, content):
                sections = extract_sections(content)
                word_count = len(content.split())

                quality_topics.append(
                    {
                        "title": title,
                        "content": content,
                        "sections": sections,
                        "word_count": word_count,
                        "source_file": str(file_path),
                    }
                )

                logger.info(
                    f"Found quality topic: {title} ({word_count} words, {len(sections)} sections)"
                )

            if processed % 50 == 0:
                logger.info(
                    f"Processed {processed} files, found {len(quality_topics)} quality topics"
                )

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue

    if len(quality_topics) < max_topics:
        logger.warning(
            f"Only found {len(quality_topics)} quality topics (requested {max_topics})"
        )

    # Shuffle and select the best topics
    random.shuffle(quality_topics)
    selected_topics = quality_topics[:max_topics]

    # Copy selected topics to target directory
    copied_count = 0

    for i, topic in enumerate(selected_topics):
        try:
            # Create clean filename
            filename = re.sub(r"[^\w\-_\(\)]", "_", topic["title"].replace(" ", "_"))
            filename = re.sub(r"_+", "_", filename).strip("_")

            if not filename:
                filename = f"topic_{i + 1}"

            # Create JSON metadata
            json_content = {
                "title": topic["title"],
                "url": f"https://en.wikipedia.org/wiki/{topic['title'].replace(' ', '_')}",
                "summary": f"Quality article about {topic['title']}",
                "sections": topic["sections"],
                "word_count": topic["word_count"],
                "source_file": topic["source_file"],
                "quality_filtered": True,
            }

            # Copy files
            json_file = json_dir / f"{filename}.json"
            txt_file = txt_dir / f"{filename}.txt"

            # Save JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)

            # Save content
            copy_file_content(topic["content"], txt_file)

            copied_count += 1
            logger.info(f"Copied: {topic['title']}")

        except Exception as e:
            logger.error(f"Failed to copy topic {topic['title']}: {e}")

    logger.info(
        f"Successfully extracted {copied_count} quality topics to {target_path}"
    )
    return copied_count > 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract quality topics from cloned FreshWiki repo"
    )
    parser.add_argument("source_dir", help="Path to cloned FreshWiki repository")
    parser.add_argument(
        "--target_dir",
        default="data/freshwiki",
        help="Target directory for quality topics",
    )
    parser.add_argument(
        "--max_topics",
        type=int,
        default=100,
        help="Number of quality topics to extract",
    )

    args = parser.parse_args()

    print("🔍 Fast Quality FreshWiki Extractor")
    print("===================================")

    success = extract_quality_topics(args.source_dir, args.target_dir, args.max_topics)

    if success:
        print("\n✅ Successully extracted quality topics!")
        print("📁 Location: {args.target_dir}")
        print("🎯 Ready or evaluation")
    else:
        print("\n❌ Extraction failed. Check the logs above.")

    # Example usage:
    # python src/utils/extract_quality_freshwiki.py /path/to/cloned/FreshWiki
