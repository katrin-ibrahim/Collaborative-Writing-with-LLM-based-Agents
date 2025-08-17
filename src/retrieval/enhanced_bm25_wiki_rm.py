"""
Enhanced BM25 Wikipedia Retrieval Manager with Lazy Loading
Uses disk-based article storage and b        db.close()

    def _populate_database(self):incrementally.
Supports full Wikipedia dumps without loading everything into memory.
"""

import sqlite3
import threading
from pathlib import Path

import logging
import os
import pickle
from rank_bm25 import BM25Okapi
from typing import Dict, List, Optional

from .data_loader import WikipediaDataLoader

logger = logging.getLogger(__name__)


class EnhancedBM25WikiRM:
    """
    Enhanced BM25 search that can handle full Wikipedia dumps efficiently.
    Uses SQLite for article storage and builds BM25 index incrementally.
    """

    def __init__(
        self,
        db_file: str = "wikipedia_bm25.db",
        bm25_cache: str = "bm25_index.pkl",
        max_articles: Optional[int] = None,
        project_root: Optional[str] = None,
        format_type: str = "rag",
    ):
        """
        Initialize enhanced BM25 retrieval manager.

        Args:
            db_file: SQLite database file for article storage
            bm25_cache: Pickle file for BM25 index cache
            max_articles: Maximum articles to index (None = all available)
            project_root: Project root directory (auto-detected if None)
        """
        logger.info("Initializing Enhanced BM25WikiRM...")

        # Determine project root
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = project_root

        logger.debug(f"Project root determined as: {self.project_root}")

        self.db_file = os.path.join(self.project_root, db_file)
        self.bm25_cache = os.path.join(self.project_root, bm25_cache)
        self.max_articles = max_articles

        logger.debug(f"Full database path: {self.db_file}")
        logger.debug(f"Database file exists: {os.path.exists(self.db_file)}")

        # Thread-local storage for database connections
        self._local = threading.local()

        # Initialize database schema and BM25 index
        self._init_database_schema()
        self.bm25, self.id_mapping, self.article_count = (
            self._load_or_build_bm25_index()
        )

        logger.info(f"Enhanced BM25WikiRM ready with {self.article_count} articles!")

    def _find_project_root(self) -> str:
        """Find the project root directory."""
        current = Path(__file__).parent.parent.parent.absolute()

        # Look for indicators of project root
        indicators = ["requirements.txt", "README.md", ".git", "src"]

        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return str(current)
            current = current.parent

        # Fallback to current directory
        return str(Path.cwd())

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "db_connection"):
            self._local.db_connection = sqlite3.connect(self.db_file)
        return self._local.db_connection

    def _init_database_schema(self):
        """Initialize SQLite database schema (thread-safe)."""
        db = sqlite3.connect(self.db_file)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                url TEXT,
                content TEXT NOT NULL,
                tokens TEXT  -- Space-separated tokens for BM25
            )
        """
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_title ON articles(title)")
        db.commit()
        db.close()

        db.close()

    def _populate_database(self) -> int:
        """Populate database with Wikipedia articles."""
        logger.info("Populating database with Wikipedia articles...")

        # Check if database is already populated
        db = self._get_db_connection()
        cursor = db.execute("SELECT COUNT(*) FROM articles")
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            logger.info(f"Database already contains {existing_count} articles")
            return existing_count

        # Load articles and populate database
        articles = WikipediaDataLoader.load_articles(self.max_articles or 1000000)

        batch_size = 1000
        total_inserted = 0

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]

            # Prepare batch data
            batch_data = []
            for article in batch:
                # Tokenize content for BM25
                tokens = article["text"].lower().split()
                tokens_str = " ".join(tokens)

                batch_data.append(
                    (article["title"], article["url"], article["text"], tokens_str)
                )

            # Insert batch
            db.executemany(
                "INSERT INTO articles (title, url, content, tokens) VALUES (?, ?, ?, ?)",
                batch_data,
            )
            db.commit()

            total_inserted += len(batch)
            if total_inserted % 10000 == 0:
                logger.info(f"Inserted {total_inserted} articles...")

        logger.info(f"Database populated with {total_inserted} articles")
        return total_inserted

    def _load_or_build_bm25_index(self):
        """Load existing BM25 index or build a new one."""
        # First, ensure database is populated
        article_count = self._populate_database()

        # Try to load cached BM25 index
        if os.path.exists(self.bm25_cache):
            logger.info("Loading cached BM25 index...")
            try:
                with open(self.bm25_cache, "rb") as f:
                    cached_data = pickle.load(f)

                # Handle both old and new cache formats
                if isinstance(cached_data, tuple) and len(cached_data) == 2:
                    bm25, id_mapping = cached_data
                    logger.info("BM25 index with ID mapping loaded from cache")
                    return bm25, id_mapping, article_count
                else:
                    # Old format without ID mapping
                    bm25 = cached_data
                    logger.info("BM25 index loaded from cache (old format)")
                    logger.info("Rebuilding to create ID mapping...")
                    # Fall through to rebuild with ID mapping

            except Exception as e:
                logger.warning(f"Failed to load BM25 cache: {e}")
                logger.info("Building new BM25 index...")

        # Build new BM25 index
        logger.info("Building BM25 index from database...")

        # Load tokenized content from database
        db = self._get_db_connection()
        cursor = db.execute("SELECT id, tokens FROM articles ORDER BY id")

        corpus = []
        id_mapping = []  # Track which database ID corresponds to each BM25 index
        batch_size = 10000

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            for article_id, tokens_str in batch:
                corpus.append(tokens_str.split())
                id_mapping.append(article_id)

            if len(corpus) % 50000 == 0:
                logger.info(f"Loaded {len(corpus)} documents for BM25...")

        # Build BM25 index
        logger.info(f"Building BM25 index with {len(corpus)} documents...")
        bm25 = BM25Okapi(corpus)

        # Cache the index and ID mapping
        logger.info("Caching BM25 index...")
        with open(self.bm25_cache, "wb") as f:
            pickle.dump((bm25, id_mapping), f)

        logger.info("BM25 index built and cached!")
        return bm25, id_mapping, len(corpus)

    def search(
        self,
        *args,
        max_results: int = None,
        format_type: str = None,
        query: str = None,
        **kwargs,
    ) -> List:
        """
        STORM-compatible search method using BM25 scoring.
        """
        if query is None or query[0] is None or query[0] == "":
            logger.debug("No query provided, returning empty results")
            return []
        # Handle STORM calling conventions (same as WikiRM)
        elif args:
            query = args[0] if len(args) > 0 else None
            if max_results is None and len(args) > 1:
                max_results = args[1]
        elif "query_or_queries" in kwargs and kwargs["query_or_queries"] is not None:
            query = kwargs.pop("query_or_queries", None)
        elif len(args) == 0 and not kwargs:
            return []

        max_results = max_results if max_results is not None else 10
        format_type = format_type if format_type else "rag"

        # Normalize input
        if isinstance(query, str):
            query_list = [query]
        else:
            query_list = list(query)

        all_results = []

        for q in query_list:
            query_results = self._search_single_query(q, max_results)
            all_results.extend(query_results)

            if len(all_results) >= max_results:
                break

        # Format results based on type
        if format_type == "rag":
            return [result["content"] for result in all_results[:max_results]]
        elif format_type == "storm":
            storm_results = []
            for result in all_results[:max_results]:
                storm_result = {
                    "snippets": result["content"],
                    "title": result["title"],
                    "url": result["url"],
                    "source": "enhanced_bm25_wiki",
                }
                storm_results.append(storm_result)
            return storm_results
        else:
            return all_results[:max_results]

    def _search_single_query(self, query: str, max_results: int) -> List[Dict]:
        """Search using BM25 scoring."""
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            if query_tokens is None or len(query_tokens) == 0:
                logger.debug("Empty query after tokenization, returning empty results")
                return []
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top results indices
            scores.argsort()
            top_indices = scores.argsort()[-max_results:][::-1]

            # Fetch articles from database
            results = []

            logger.debug(f"Search method using database file: {self.db_file}")

            # Use a single connection for all queries to avoid connection issues
            import sqlite3

            import os

            logger.debug(f"Opening single connection to: {self.db_file}")
            logger.debug(f"File exists: {os.path.exists(self.db_file)}")
            logger.debug(
                f"File size: {os.path.getsize(self.db_file) if os.path.exists(self.db_file) else 'N/A'}"
            )

            db = sqlite3.connect(self.db_file)

            # Test the connection
            cursor = db.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            test_count = cursor.fetchone()[0]
            logger.debug(f"Connection test: {test_count} articles found")

            # Debug: Check table structure
            cursor.execute("PRAGMA table_info(articles)")
            columns = cursor.fetchall()
            logger.debug(f"Table structure: {[col[1] for col in columns]}")

            # Debug: Test a specific known ID first
            test_id = 127
            cursor.execute("SELECT title FROM articles WHERE id = ?", (test_id,))
            test_row = cursor.fetchone()
            logger.debug(f"Test query for ID {test_id}: {test_row is not None}")
            if test_row:
                logger.debug(f"Test title: {test_row[0]}")

            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    # Use ID mapping to get the correct database ID
                    db_id = self.id_mapping[idx]

                    logger.debug(
                        f"Processing BM25 index {idx} -> DB ID {db_id}, score: {scores[idx]}"
                    )

                    # Fetch article from database using the same connection
                    cursor.execute(
                        "SELECT title, url, content FROM articles WHERE id = ?",
                        (db_id,),
                    )
                    row = cursor.fetchone()

                    logger.debug(f"Query result for ID {db_id}: {row is not None}")

                    if row:
                        title, url, content = row
                        result_dict = {
                            "content": content,
                            "title": title,
                            "url": url,
                            "source": "enhanced_bm25_wiki",
                            "score": float(scores[idx]),
                        }
                        logger.debug(f"Created result dict: {type(result_dict)}")
                        logger.debug(f"Result dict keys: {list(result_dict.keys())}")
                        logger.debug(f"Result dict title: {result_dict['title']}")
                        results.append(result_dict)
                        logger.debug(f"Results list now has {len(results)} items")
                        logger.debug(f"Last added item type: {type(results[-1])}")
                        logger.debug(f"Added result: {title}")
                    else:
                        logger.warning(
                            f"No article found for BM25 index {idx} (DB ID {db_id})"
                        )

            # Close the database connection
            db.close()

            logger.debug(f"About to return results: {len(results)} items")
            for i, item in enumerate(results):
                logger.debug(f"Result {i} type: {type(item)}")
                if isinstance(item, dict):
                    logger.debug(f"Result {i} title: {item.get('title', 'NO TITLE')}")
                else:
                    logger.debug(f"Result {i} content (first 50): {str(item)[:50]}")

            # Return a copy to avoid any reference issues
            final_results = [
                dict(item) if isinstance(item, dict) else item for item in results
            ]
            logger.debug(f"Final results copy: {len(final_results)} items")
            for i, item in enumerate(final_results):
                logger.debug(f"Final result {i} type: {type(item)}")

            return final_results

        except Exception as e:
            logger.error(f"Enhanced BM25 search failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get statistics about the loaded index."""
        return {
            "num_articles": self.article_count,
            "db_file": self.db_file,
            "bm25_cache": self.bm25_cache,
            "db_size_mb": (
                os.path.getsize(self.db_file) / (1024 * 1024)
                if os.path.exists(self.db_file)
                else 0
            ),
        }

    def close(self):
        """Clean up database connections."""
        if hasattr(self, "_local") and hasattr(self._local, "db_connection"):
            self._local.db_connection.close()
            delattr(self._local, "db_connection")

    def __call__(self, *args, **kwargs):
        """Support callable interface for STORM compatibility."""
        return self.search(*args, **kwargs)
