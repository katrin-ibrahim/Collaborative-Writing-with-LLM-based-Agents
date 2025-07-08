# src/evaluation/metrics/heading_metrics.py
import re
from typing import List


class HeadingMetrics:
    """
    Heading analysis using semantic similarity.
    Uses sentence transformers for better semantic matching without hardcoded synonyms.
    """

    def __init__(self):
        self.embedder = None
        self._init_embedder()

    def _init_embedder(self):
        """Initialize sentence transformer with fallback to word overlap."""
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.np = np

        except ImportError:
            # Fallback to word overlap if sentence-transformers not available
            self.embedder = None

    def extract_headings_from_content(self, content: str) -> List[str]:
        """Extract headings from markdown-style content."""
        headings = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                # Remove markdown symbols and extract heading text
                heading = re.sub(r"^#+\s*", "", line).strip()
                if heading and len(heading) > 1:
                    headings.append(heading)

        return headings

    def normalize_heading(self, heading: str) -> str:
        """Basic heading normalization."""
        # Remove numbering and extra whitespace
        heading = re.sub(r"^\d+\.?\s*", "", heading)
        heading = " ".join(heading.split())
        return heading.strip()

    def calculate_semantic_similarity(self, heading1: str, heading2: str) -> float:
        """
        Calculate semantic similarity between headings.
        Uses sentence transformers if available, fallback to word overlap.
        """
        norm1 = self.normalize_heading(heading1)
        norm2 = self.normalize_heading(heading2)

        if self.embedder is not None:
            # Use semantic similarity
            try:
                embeddings = self.embedder.encode([norm1, norm2])
                similarity = self.np.dot(embeddings[0], embeddings[1]) / (
                    self.np.linalg.norm(embeddings[0])
                    * self.np.linalg.norm(embeddings[1])
                )
                return max(0.0, float(similarity))
            except:
                # Fall back to word overlap if embedding fails
                pass

        # Fallback: simple word overlap
        words1 = set(norm1.lower().split())
        words2 = set(norm2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def calculate_heading_soft_recall(
        self, generated_headings: List[str], reference_headings: List[str]
    ) -> float:
        """
        Calculate Heading Soft Recall (HSR) using semantic similarity.

        For each reference heading, finds the best matching generated heading
        and averages the similarity scores.
        """
        if not reference_headings:
            return 1.0

        if not generated_headings:
            return 0.0

        total_similarity = 0.0

        for ref_heading in reference_headings:
            best_similarity = 0.0

            for gen_heading in generated_headings:
                similarity = self.calculate_semantic_similarity(
                    ref_heading, gen_heading
                )
                best_similarity = max(best_similarity, similarity)

            total_similarity += best_similarity

        return total_similarity / len(reference_headings)
