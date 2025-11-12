# src/utils/content/description_generator.py

import re
from typing import List, Optional


class DescriptionGenerator:
    """Generate consistent, informative descriptions for research chunks across all RMs."""

    @staticmethod
    def extract_substantive_content(
        content: str,
        max_length: int = 150,
        min_line_length: int = 30,
        skip_patterns: Optional[List[str]] = None,
    ) -> str:
        """
        Extract meaningful content, skipping headers and markup.

        Args:
            content: The text content to extract from
            max_length: Maximum length of extracted content
            min_line_length: Minimum line length to consider substantive
            skip_patterns: Additional patterns to skip (case-insensitive)

        Returns:
            Cleaned, substantive content preview
        """
        if not content:
            return ""

        # Default skip patterns (common across Wikipedia, academic papers, etc.)
        default_skip_patterns = [
            "see also",
            "references",
            "external links",
            "further reading",
            "category:",
            "file:",
            "thumb|",
            "image:",
            "notes",
            "bibliography",
            "citations",
            "works cited",
        ]

        skip_patterns = skip_patterns or []
        all_skip_patterns = default_skip_patterns + [p.lower() for p in skip_patterns]

        lines = content.split("\n")
        content_lines = []

        # Filter out headers, short lines, and navigation text
        for line in lines:
            line = line.strip()

            # Skip if too short
            if len(line) < min_line_length:
                continue

            # Skip headers
            if line.startswith("#") or line.startswith("="):
                continue

            # Skip wiki/markdown markup
            if "{{" in line or "[[" in line:
                continue

            # Skip navigation patterns
            if any(line.lower().startswith(pattern) for pattern in all_skip_patterns):
                continue

            content_lines.append(line)

        if not content_lines:
            # Fallback: use original content but clean it
            clean_content = content.replace("\n", " ").strip()
            # Remove common markup
            clean_content = re.sub(r"\[\[|\]\]", "", clean_content)  # Wiki links
            clean_content = re.sub(r"'''|''", "", clean_content)  # Wiki bold/italic
            clean_content = re.sub(r"\{\{.*?\}\}", "", clean_content)  # Wiki templates
            clean_content = re.sub(r"\s+", " ", clean_content)  # Normalize whitespace
            return (
                clean_content[:max_length] + "..."
                if len(clean_content) > max_length
                else clean_content
            )

        # Take the first substantial content line
        first_content = content_lines[0]

        # If it's very long, trim to sentence boundary
        if len(first_content) > max_length:
            sentences = re.split(r"[.!?]+", first_content)
            first_content = sentences[0].strip()

            # If first sentence is too short, add second sentence
            if len(first_content) < max_length * 0.6 and len(sentences) > 1:
                second_sentence = sentences[1].strip()
                if second_sentence:
                    first_content += ". " + second_sentence

        # Final cleanup
        first_content = first_content.strip()
        if len(first_content) > max_length:
            first_content = first_content[: max_length - 3] + "..."

        return first_content

    @staticmethod
    def determine_chunk_position(chunk_idx: int, total_chunks: int) -> str:
        """
        Determine human-readable position of chunk in document.

        Args:
            chunk_idx: Zero-based index of current chunk
            total_chunks: Total number of chunks in document

        Returns:
            Position description (e.g., "introduction", "middle section")
        """
        if total_chunks == 1:
            return "complete document"
        elif chunk_idx == 0:
            return "introduction"
        elif chunk_idx == total_chunks - 1:
            return "conclusion"
        elif chunk_idx < total_chunks / 3:
            return "early section"
        elif chunk_idx < 2 * total_chunks / 3:
            return "middle section"
        else:
            return "later section"

    @staticmethod
    def create_description(
        content: str,
        title: str,
        chunk_idx: int = 0,
        total_chunks: int = 1,
        categories: Optional[List[str]] = None,
        max_preview_length: int = 200,
        include_title: bool = False,
        include_position: bool = False,
        include_categories: bool = False,
    ) -> str:
        """
        Create a comprehensive, informative description for a research chunk.

        Args:
            content: The chunk content text
            title: Document/article title
            chunk_idx: Zero-based index of this chunk
            total_chunks: Total chunks in the document
            categories: Optional list of categories/topics
            max_preview_length: Maximum length for content preview
            include_title: Whether to include the title
            include_position: Whether to include chunk position info
            include_categories: Whether to include category information

        Returns:
            Formatted description string
        """
        # Extract substantive content preview
        preview = DescriptionGenerator.extract_substantive_content(
            content, max_length=max_preview_length
        )

        # Build description part
        parts = []
        if include_title:
            parts.append(title)

        # Categories (if available and requested)
        if include_categories and categories:
            category_str = ", ".join(categories[:2])  # Limit to 2 categories
            parts.append(f"[{category_str}]")

        # Metadata (if available) - e.g., year, authors

        # Position information (if requested and multi-chunk)
        if include_position and total_chunks > 1:
            position = DescriptionGenerator.determine_chunk_position(
                chunk_idx, total_chunks
            )
            parts.append(f"- {position} (chunk {chunk_idx + 1}/{total_chunks})")
        elif include_position and total_chunks == 1:
            parts.append("- complete document")

        # Join parts
        description = " ".join(parts)

        # Add preview if available
        if preview:
            description += f": {preview}"

        return description
