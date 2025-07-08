import logging
from typing import List

from utils.data_models import SearchResult

logger = logging.getLogger(__name__)


class PassageRanker:
    """Ranks and selects top-k passages from search results."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def rank_passages(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """
        Rank passages by relevance to the query.

        In a full implementation, this would use sophisticated ranking
        algorithms like BM25, neural retrievers, or cross-encoders.
        """
        # Sort by relevance score (already provided by search engine)
        ranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)

        # Return top-k results
        top_results = ranked_results[: self.top_k]

        logger.info(f"Ranked and selected top {len(top_results)} passages")
        return top_results

    def create_context(self, passages: List[SearchResult]) -> str:
        """Combine top passages into a working context."""
        context_parts = []

        for i, passage in enumerate(passages, 1):
            context_parts.append(f"[Source {i}]: {passage.content}")

        full_context = "\n\n".join(context_parts)
        logger.info(f"Created context with {len(passages)} passages")

        return full_context
