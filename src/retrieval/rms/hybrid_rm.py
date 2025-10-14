# src/retrieval/rms/hybrid_rm.py
"""
Hybrid retrieval manager combining WikiRM (live data) with FAISS (broad coverage).
"""

import logging
from typing import Dict, List

from src.retrieval.rms.base_retriever import BaseRetriever
from src.retrieval.rms.supabase_faiss_rm import FaissRM
from src.retrieval.rms.wiki_rm import WikiRM

logger = logging.getLogger(__name__)


class HybridRM(BaseRetriever):
    """
    Hybrid retrieval manager that combines:
    - WikiRM: Live Wikipedia data for current, specific information
    - FAISS: Broad coverage from cached embeddings for comprehensive context

    Strategy:
    1. Query WikiRM first for live, specific results (high precision)
    2. Query FAISS for additional context and coverage (high recall)
    3. Merge and deduplicate results
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize both retrieval managers
        try:
            self.wiki_rm = WikiRM(**kwargs)
            logger.info("WikiRM initialized successfully")
        except Exception as e:
            logger.warning(f"WikiRM initialization failed: {e}")
            self.wiki_rm = None

        try:
            self.faiss_rm = FaissRM(**kwargs)
            logger.info("FAISS RM initialized successfully")
        except Exception as e:
            logger.warning(f"FAISS RM initialization failed: {e}")
            self.faiss_rm = None

        if not self.wiki_rm and not self.faiss_rm:
            raise RuntimeError("Both WikiRM and FAISS RM failed to initialize")

        # Configuration for hybrid behavior - balanced approach (optimal from evaluation)
        self.wiki_weight = 0.5  # Equal weight for current/specific info
        self.faiss_weight = 0.5  # Equal weight for broad coverage
        self.max_wiki_results = 6  # Balanced results from wiki
        self.max_faiss_results = 6  # Balanced results from faiss

    def _retrieve_article(
        self, query: str, topic: str = None, max_results: int = None
    ) -> List[Dict]:
        """
        Retrieve articles using hybrid approach - required by BaseRetriever.
        This method is called by the base class search method.
        """
        wiki_results = []
        faiss_results = []

        # 1. Query WikiRM for live, specific data
        if self.wiki_rm:
            try:
                wiki_results = self.wiki_rm._retrieve_article(
                    query=query, topic=topic, max_results=self.max_wiki_results
                )
                logger.info(f"WikiRM returned {len(wiki_results)} results")
            except Exception as e:
                logger.warning(f"WikiRM search failed: {e}")

        # 2. Query FAISS for broader coverage
        if self.faiss_rm:
            try:
                faiss_results = self.faiss_rm._retrieve_article(
                    query=query, topic=topic, max_results=self.max_faiss_results
                )
                logger.info(f"FAISS RM returned {len(faiss_results)} results")
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")

        # 3. Merge and rank results
        combined_results = self._merge_and_rank_results(
            wiki_results, faiss_results, query
        )

        logger.info(f"Hybrid search combined to {len(combined_results)} total results")
        return combined_results

    def _merge_and_rank_results(
        self, wiki_results: List[Dict], faiss_results: List[Dict], query: str
    ) -> List[Dict]:
        """
        Merge and rank results from both retrieval managers.

        Strategy:
        1. Add source tags to identify origin
        2. Apply weighting based on source reliability
        3. Deduplicate similar content
        4. Sort by weighted relevance score
        """
        merged_results = []

        # Process WikiRM results (higher weight)
        for i, result in enumerate(wiki_results):
            result_copy = result.copy()
            result_copy["hybrid_source"] = "wiki"
            result_copy["hybrid_rank"] = i + 1

            # Boost wiki scores
            original_score = result_copy.get("relevance_score", 0.5)
            result_copy["relevance_score"] = original_score * self.wiki_weight + 0.3

            merged_results.append(result_copy)

        # Process FAISS results (lower weight)
        for i, result in enumerate(faiss_results):
            result_copy = result.copy()
            result_copy["hybrid_source"] = "faiss"
            result_copy["hybrid_rank"] = i + 1

            # Apply faiss weight (generally lower)
            original_score = result_copy.get("relevance_score", 0.5)
            result_copy["relevance_score"] = original_score * self.faiss_weight + 0.1

            # Check for duplicates (simple content similarity)
            if not self._is_duplicate(result_copy, merged_results):
                merged_results.append(result_copy)
            else:
                logger.debug("Skipped duplicate FAISS result")

        # Sort by weighted relevance score (higher is better)
        merged_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return merged_results

    def _is_duplicate(self, new_result: Dict, existing_results: List[Dict]) -> bool:
        """
        Simple duplicate detection based on content similarity.
        """
        new_content = str(new_result.get("content", "")).lower()
        new_title = str(new_result.get("title", "")).lower()

        # Skip very short content
        if len(new_content) < 50:
            return False

        for existing in existing_results:
            existing_content = str(existing.get("content", "")).lower()
            existing_title = str(existing.get("title", "")).lower()

            # Check title similarity
            if new_title and existing_title:
                if new_title in existing_title or existing_title in new_title:
                    if (
                        len(max(new_title, existing_title)) > 10
                    ):  # Avoid short title false positives
                        return True

            # Check content overlap (simple substring approach)
            if len(new_content) > 100 and len(existing_content) > 100:
                # Check if significant portion overlaps
                shorter = min(new_content, existing_content, key=len)
                longer = max(new_content, existing_content, key=len)

                if len(shorter) > 50 and shorter[:100] in longer:
                    return True

        return False

    def get_config_info(self) -> Dict:
        """Get configuration information for debugging."""
        return {
            "type": "hybrid",
            "wiki_rm_available": self.wiki_rm is not None,
            "faiss_rm_available": self.faiss_rm is not None,
            "wiki_weight": self.wiki_weight,
            "faiss_weight": self.faiss_weight,
            "max_wiki_results": self.max_wiki_results,
            "max_faiss_results": self.max_faiss_results,
        }
