"""
Relevance scoring utilities for content ranking and similarity.
"""

import logging
import numpy as np
from numpy.linalg import norm
from typing import Any, Dict, List, Union

try:
    from sentence_transformers import SentenceTransformer

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
        self.similarity_threshold = similarity_threshold
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def calculate_relevance(
        self,
        query: str,
        content: Union[List[str], List[Dict[str, Any]]],
        threshold: float = None,
        max_items: int = None,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Calculate semantic relevance between query and content.

        Args:
            query: Query string to compare against
            content: List of content strings OR list of result dictionaries
            threshold: Minimum similarity score to keep (uses instance default if None)
            max_items: Maximum number of items to return

        Returns:
            Filtered and sorted content by relevance
        """
        if not content or not query.strip():
            return content

        threshold = threshold if threshold is not None else self.similarity_threshold

        # Handle both string lists and dict lists
        is_dict_input = isinstance(content[0], dict) if content else False

        if is_dict_input:
            return self._score_dict_results(query, content, threshold, max_items)
        else:
            return self._score_string_content(query, content, threshold, max_items)

    def _score_string_content(
        self, query: str, content_items: List[str], threshold: float, max_items: int
    ) -> List[str]:
        """Score and filter string content items."""
        similarities = self._calculate_similarities(query, content_items)

        # Filter by threshold and sort by relevance
        relevant_items = [item for item, score in similarities if score >= threshold]

        if max_items:
            relevant_items = relevant_items[:max_items]

        return relevant_items

    def _score_dict_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        threshold: float,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        """Score and filter dictionary results."""
        # Extract content for scoring
        contents = [result.get("snippets", "") for result in results]
        # verify contents is a list of strings
        contents = [
            (
                result.get("snippets", "")
                if isinstance(result.get("snippets"), str)
                else (
                    " ".join(result.get("snippets", []))
                    if isinstance(result.get("snippets"), list)
                    else str(result.get("snippets", ""))
                )
            )
            for result in results
        ]

        # Filter out empty strings
        contents = [content for content in contents if content and content.strip()]

        # Calculate similarities
        similarities = self._calculate_similarities(query, contents)

        # Add relevance scores to results and filter
        scored_results = []
        for result, (_, score) in zip(results, similarities):
            if score >= threshold:
                result_copy = result.copy()
                result_copy["topic_relevance"] = score
                scored_results.append(result_copy)

        # Sort by relevance score
        scored_results.sort(key=lambda x: x.get("topic_relevance", 0), reverse=True)

        if max_items:
            scored_results = scored_results[:max_items]

        return scored_results

    def _calculate_similarities(
        self, query: str, content_items: List[str]
    ) -> List[tuple]:
        """
        Calculate semantic similarities between query and content items.

        Args:
            query: Query string to compare against
            content_items: List of content strings to score

        Returns:
            List of (content, similarity_score) tuples sorted by score (highest first)
        """
        if not content_items or not query.strip():
            return []

        # Debug: Check for problematic strings
        print(f"DEBUG: content_items length: {len(content_items)}")

        try:
            query_embedding = self.embedding_model.encode([query])
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]

            # Try encoding with filtered items
            item_embeddings = self.embedding_model.encode(content_items)
        except Exception as e:
            logger.error(f"Failed to encode content items: {e}")
            return []

        # Calculate similarities
        similarities = []
        for item, embedding in zip(content_items, item_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                norm(query_embedding) * norm(embedding)
            )
            similarities.append((item, float(similarity)))

        # Sort by similarity score (highest first)
        return sorted(similarities, key=lambda x: x[1], reverse=True)
