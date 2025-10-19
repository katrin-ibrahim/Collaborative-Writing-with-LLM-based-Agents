"""
Relevance scoring utilities for content ranking and similarity.
"""

import logging
import numpy as np
from numpy.linalg import norm
from typing import Any, Dict, List, Optional, Tuple

from src.config.config_context import ConfigContext

try:
    pass

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """Handles semantic similarity and relevance scoring operations."""

    def __init__(self, similarity_threshold: float = 0.3):
        """
        Initialize relevance scorer.

        Args:
            similarity_threshold: Minimum similarity score to consider relevant
        """
        retrieval_config = ConfigContext.get_retrieval_config()
        if retrieval_config is None:
            raise RuntimeError(
                "Retrieval config is None. Ensure ConfigContext is properly initialized before using RelevanceScorer."
            )
        self.similarity_threshold = similarity_threshold
        embedding_model_name = retrieval_config.embedding_model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer(
                    embedding_model_name, device="cpu"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None

    def calculate_relevance(
        self,
        query: str,
        content: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calculate semantic relevance between query and content results.

        This function assumes the input and output is a list of dictionary results,
        which matches the data structure used by the BaseRetriever.

        Args:
            query: Query string to compare against
            content: List of search result dictionaries
            threshold: Minimum similarity score to keep (uses instance default if None)
            max_items: Maximum number of items to return

        Returns:
            Filtered and sorted search result dictionaries by relevance
        """
        if not content or not query.strip():
            return content

        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            logger.warning(
                "Embedding model not available or loaded, skipping relevance scoring."
            )
            return content

        threshold = threshold if threshold is not None else self.similarity_threshold

        # Explicitly call dict scoring as the signature now enforces this type
        max_items_int = max_items if max_items is not None else 0
        return self._score_results(query, content, threshold, max_items_int)

    def _score_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        threshold: float,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        """Score and filter dictionary results."""

        # 1. Extract content and map back to original results
        contents = []
        filtered_results_map: Dict[str, Dict[str, Any]] = {}

        for result in results:
            raw_snippets = result.get("snippets", "") or result.get("content", "")

            # Ensure snippets are combined into a single string for encoding
            if isinstance(raw_snippets, list):
                content_str = " ".join(str(s) for s in raw_snippets if s)
            else:
                content_str = str(raw_snippets)

            if content_str and content_str.strip():
                contents.append(content_str)
                # Map content string back to original result dictionary for easy retrieval
                # Note: This relies on content_str being unique enough for mapping
                filtered_results_map[content_str] = result

        if not contents:
            return []

        # 2. Calculate similarities on the extracted content strings
        # Similarities is List[tuple[content_str, float]]
        similarities = self.calculate_similarities(query, contents)

        # 3. Add relevance scores to results and filter
        scored_results = []
        for content_str, score in similarities:
            if score >= threshold:
                # Retrieve the original result dict using the content string
                result_copy = filtered_results_map[content_str].copy()
                result_copy["topic_relevance"] = score
                scored_results.append(result_copy)

        # 4. Sort and limit
        scored_results.sort(key=lambda x: x.get("topic_relevance", 0), reverse=True)

        if max_items:
            scored_results = scored_results[:max_items]

        return scored_results

    def calculate_similarities(
        self, query: str, content_items: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Calculate semantic similarities between query and content items.

        Args:
            query: Query string to compare against
            content_items: List of content strings to score

        Returns:
            List of (content_string, similarity_score) tuples sorted by score (highest first)
        """
        if not content_items or not query.strip():
            return []

        if self.embedding_model is None:
            logger.error("Embedding model is None, cannot calculate similarities.")
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]

            item_embeddings = self.embedding_model.encode(content_items)
        except Exception as e:
            logger.error(f"Failed to encode content items: {e}")
            return []

        if not isinstance(item_embeddings, np.ndarray) or item_embeddings.size == 0:
            logger.error("Failed to generate valid embeddings for content items.")
            return []

        # Calculate similarities
        similarities = []
        for item, embedding in zip(content_items, item_embeddings):
            # Cosine similarity
            q_norm = norm(query_embedding)
            e_norm = norm(embedding)

            if q_norm == 0 or e_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_embedding, embedding) / (q_norm * e_norm)

            similarities.append((item, float(similarity)))

        # Sort by similarity score (highest first)
        return sorted(similarities, key=lambda x: x[1], reverse=True)
