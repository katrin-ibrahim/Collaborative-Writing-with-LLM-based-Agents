# src/evaluation/metrics/heading_metrics.py
import logging
import numpy as np
import re
from typing import List

logger = logging.getLogger(__name__)


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
            logger.error("Sentence transformers not available.")

    def extract_headings_from_content(self, content: str) -> List[str]:
        """Extract headings from markdown-style content."""
        headings = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                heading = re.sub(r"^#+\s*", "", line).strip()
                if heading and len(heading) > 1:
                    headings.append(heading)
        logger.info(f"Extracted {len(headings)} headings from content")
        return headings

    def calculate_heading_soft_recall(
        self, generated_headings: List[str], reference_headings: List[str]
    ) -> float:
        """
        Calculate Heading Soft Recall (HSR) using STORM's soft recall cardinality formula.

        Formula: HSR = card(G ∩ P) / card(G)
        Where: card(A) = Σ count(Ai) and count(Ai) = 1 / Σ Sim(Ai, Aj)
        """
        logger.info("=== Heading Soft Recall Calculation ===")
        logger.info(f"Generated headings: {generated_headings}")
        logger.info(f"Reference headings: {reference_headings}")
        if not reference_headings or not generated_headings:
            logger.warning(
                f"No headings provided, returning 0.0, len(generated_headings): {len(generated_headings)}, len(reference_headings): {len(reference_headings)}"
            )
            return 0.0

        # Get embeddings for all headings
        ref_embeddings = self.embedder.encode(reference_headings)
        gen_embeddings = self.embedder.encode(generated_headings)

        # Calculate soft cardinality for reference headings (G)
        ref_card = self._calculate_soft_cardinality(ref_embeddings)

        # Calculate soft cardinality for generated headings (P)
        gen_card = self._calculate_soft_cardinality(gen_embeddings)

        # Calculate soft cardinality for union (G ∪ P)
        all_embeddings = self.np.vstack([ref_embeddings, gen_embeddings])
        union_card = self._calculate_soft_cardinality(all_embeddings)

        # Raw intersection via inclusion-exclusion
        raw_intersection = ref_card + gen_card - union_card
        # Clamp to valid range: [0, min(ref_card, gen_card)]
        intersection_card = max(0.0, min(raw_intersection, ref_card, gen_card))

        # HSR = card(G ∩ P) / card(G)
        hsr = intersection_card / ref_card if ref_card > 0 else 0.0

        return hsr

    def _calculate_soft_cardinality(self, embeddings: np.ndarray) -> float:
        """
        Calculate soft cardinality using STORM's formula.
        card(A) = Σ count(Ai) where count(Ai) = 1 / Σ Sim(Ai, Aj)
        """
        if len(embeddings) == 0:
            logger.warning(
                "Empty embeddings provided, returning 0.0 for soft cardinality"
            )
            return 0.0

        # Calculate cosine similarity matrix
        similarity_matrix = self.np.dot(embeddings, embeddings.T)

        soft_cardinality = 0.0
        for i in range(len(embeddings)):
            sim_sum = self.np.sum(similarity_matrix[i])
            count_i = 1.0 / sim_sum if sim_sum > 0 else 0.0
            soft_cardinality += count_i

        logger.info(f"Soft cardinality calculated: {soft_cardinality}")
        return soft_cardinality
