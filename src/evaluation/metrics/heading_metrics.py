# src/evaluation/metrics/heading_metrics.py
import numpy as np
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

            self.embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
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

    # def calculate_heading_soft_recall(
    #     self, generated_headings: List[str], reference_headings: List[str]
    # ) -> float:
    #     """
    #     Calculate Heading Soft Recall (HSR) using semantic similarity.

    #     For each reference heading, finds the best matching generated heading
    #     and averages the similarity scores.
    #     """
    #     if not reference_headings:
    #         return 1.0

    #     if not generated_headings:
    #         return 0.0

    #     total_similarity = 0.0

    #     for ref_heading in reference_headings:
    #         best_similarity = 0.0

    #         for gen_heading in generated_headings:
    #             similarity = self.calculate_semantic_similarity(
    #                 ref_heading, gen_heading
    #             )
    #             best_similarity = max(best_similarity, similarity)

    #         total_similarity += best_similarity

    #     return total_similarity / len(reference_headings)
    def calculate_heading_soft_recall(
        self, generated_headings: List[str], reference_headings: List[str]
    ) -> float:
        """
        Calculate Heading Soft Recall (HSR) using STORM's soft recall cardinality formula.

        Formula: HSR = card(G ∩ P) / card(G)
        Where: card(A) = Σ count(Ai) and count(Ai) = 1 / Σ Sim(Ai, Aj)
        """
        if not reference_headings:
            return 1.0

        if not generated_headings:
            return 0.0

        # Get embeddings for all headings
        ref_embeddings = self.embedder.encode(reference_headings)
        gen_embeddings = self.embedder.encode(generated_headings)

        # Calculate soft cardinality for reference headings (G)
        ref_card = self._calculate_soft_cardinality(ref_embeddings)

        # Calculate soft cardinality for generated headings (P)
        gen_card = self._calculate_soft_cardinality(gen_embeddings)

        # Calculate soft cardinality for union (G ∪ P)
        all_embeddings = np.vstack([ref_embeddings, gen_embeddings])
        union_card = self._calculate_soft_cardinality(all_embeddings)

        # Apply soft recall formula: card(G ∩ P) = card(G) + card(P) - card(G ∪ P)
        intersection_card = ref_card + gen_card - union_card

        # HSR = card(G ∩ P) / card(G)
        hsr = intersection_card / ref_card if ref_card > 0 else 0.0

        return max(0.0, min(1.0, hsr))  # Clamp to [0,1]

    def _calculate_soft_cardinality(self, embeddings: np.ndarray) -> float:
        """
        Calculate soft cardinality using STORM's formula.
        card(A) = Σ count(Ai) where count(Ai) = 1 / Σ Sim(Ai, Aj)
        """
        if len(embeddings) == 0:
            return 0.0

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)

        soft_cardinality = 0.0
        for i in range(len(embeddings)):
            # Sum of similarities for embedding i with all embeddings
            sim_sum = np.sum(similarity_matrix[i])
            # count(Ai) = 1 / sim_sum
            count_i = 1.0 / sim_sum if sim_sum > 0 else 0.0
            soft_cardinality += count_i

        return soft_cardinality
