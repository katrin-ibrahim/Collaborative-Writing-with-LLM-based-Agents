"""
BM25 Wikipedia Retrieval Manager
Uses BM25 scoring over Wikipedia articles from local XML dumps.
Good for exact entity matching and keyword search.
"""

import logging
from rank_bm25 import BM25Okapi
from typing import Dict, List

from .data_loader import WikipediaDataLoader

logger = logging.getLogger(__name__)


class BM25WikiRM:
    """
    BM25 search over Wikipedia articles from local XML dumps.
    Good for exact entity matching and keyword search.
    """

    def __init__(self, num_articles: int = 100000):
        logger.info("Initializing BM25WikiRM...")

        # Load articles
        self.articles = WikipediaDataLoader.load_articles(num_articles)

        # Build BM25 index
        logger.info("Building BM25 index...")
        corpus = [article["text"].lower().split() for article in self.articles]
        self.bm25 = BM25Okapi(corpus)

        logger.info(f"BM25WikiRM ready with {len(self.articles)} articles!")

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search using BM25 scoring."""
        try:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)

            # Get top results
            top_indices = scores.argsort()[-max_results:][::-1]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    article = self.articles[idx]
                    results.append(
                        {
                            "content": article["text"],
                            "title": article["title"],
                            "url": article["url"],
                            "source": "bm25_wiki",
                            "score": float(scores[idx]),
                        }
                    )

            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
