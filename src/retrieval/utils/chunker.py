"""
Content chunking utilities for text processing.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class ContentChunker:
    """Handles text chunking for large content."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize content chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving paragraph boundaries.
        If a paragraph exceeds chunk_size, split it at sentence boundaries (periods).

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If single paragraph is larger than chunk size, split at periods
                if len(paragraph) > self.chunk_size:
                    sentences = paragraph.split(". ")
                    temp_chunk = ""

                    for i, sentence in enumerate(sentences):
                        # Add period back except for last sentence
                        if i < len(sentences) - 1:
                            sentence += "."

                        # If adding this sentence would exceed chunk size
                        if len(temp_chunk) + len(sentence) > self.chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sentence
                            else:
                                # Single sentence is too long, just add it anyway
                                chunks.append(sentence.strip())
                        else:
                            if temp_chunk:
                                temp_chunk += " " + sentence
                            else:
                                temp_chunk = sentence

                    # Add remaining content as current chunk
                    if temp_chunk:
                        current_chunk = temp_chunk.strip()
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
