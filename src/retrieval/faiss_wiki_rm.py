"""
FAISS Wikipedia Retrieval Manager
Uses FAISS semantic search over Wikipedia articles from local XML dumps.
Good for finding semantically related content and entity relationships.
"""

import faiss
import logging
import os
import pickle
from sentence_transformers import SentenceTransformer
from typing import Dict, List

from .data_loader import WikipediaDataLoader

logger = logging.getLogger(__name__)


class FAISSWikiRM:
    """
    FAISS semantic search over Wikipedia articles from local XML dumps.
    Good for finding semantically related content and entity relationships.
    """

    def __init__(
        self, num_articles: int = 100000, embedding_model: str = "all-MiniLM-L6-v2"
    ):
        logger.info("Initializing FAISSWikiRM...")

        self.embedding_model_name = embedding_model
        self.embeddings_cache = f"faiss_embeddings_{num_articles}.pkl"
        self.index_cache = f"faiss_index_{num_articles}.index"

        # Load articles
        self.articles = WikipediaDataLoader.load_articles(num_articles)

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        # Build or load FAISS index
        self.index = self._build_or_load_index()

        logger.info(f"FAISSWikiRM ready with {len(self.articles)} articles!")

    def _build_or_load_index(self):
        """Build FAISS index or load from cache."""
        if os.path.exists(self.index_cache) and os.path.exists(self.embeddings_cache):
            logger.info("Loading cached FAISS index...")
            index = faiss.read_index(self.index_cache)
            return index

        logger.info("Building FAISS index (this may take a while)...")

        # Create embeddings for all articles
        texts = [
            article["text"][:1000] for article in self.articles
        ]  # Truncate for speed
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Save embeddings cache
        with open(self.embeddings_cache, "wb") as f:
            pickle.dump(embeddings, f)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype("float32"))

        # Save index
        faiss.write_index(index, self.index_cache)

        logger.info("FAISS index built and cached!")
        return index

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search using FAISS semantic similarity."""
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.astype("float32"), max_results
            )

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    article = self.articles[idx]
                    results.append(
                        {
                            "content": article["text"],
                            "title": article["title"],
                            "url": article["url"],
                            "source": "faiss_wiki",
                            "score": float(score),
                        }
                    )

            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
