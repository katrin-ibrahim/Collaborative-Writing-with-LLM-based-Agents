"""
Enhanced BM25 Wikipedia Retrieval Manager with Advanced Text Preprocessing
Uses robust text preprocessing, proper tokenization, and Wikipedia content cleaning.
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
from .text_preprocessor import EnhancedTextPreprocessor

logger = logging.getLogger(__name__)


class EnhancedBM25WikiRM:
    """
    Enhanced BM25 search with advanced text preprocessing and Wikipedia content cleaning.
    Uses SQLite for article storage and builds BM25 index with proper tokenization.
    """

    def __init__(
        self,
        db_file: str = "wikipedia_bm25.db",
        bm25_cache: str = "bm25_index.pkl",
        max_articles: Optional[int] = None,
        project_root: Optional[str] = None,
        format_type: str = "rag",
        use_stemming: bool = True,
        language: str = "english",
    ):
        """
        Initialize enhanced BM25 retrieval manager with advanced preprocessing.

        Args:
            db_file: SQLite database file for article storage
            bm25_cache: Pickle file for BM25 index cache
            max_articles: Maximum articles to index (None = all available)
            project_root: Project root directory (auto-detected if None)
            format_type: Default format type for results
            use_stemming: Whether to use stemming in preprocessing
            language: Language for stop words and processing
        """
        logger.info("Initializing Enhanced BM25WikiRM with advanced preprocessing...")

        # Determine project root
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = project_root

        logger.debug(f"Project root determined as: {self.project_root}")

        self.db_file = os.path.join(self.project_root, db_file)
        self.bm25_cache = os.path.join(self.project_root, bm25_cache)
        self.max_articles = max_articles

        # Initialize advanced text preprocessor
        self.preprocessor = EnhancedTextPreprocessor(
            language=language, use_stemming=use_stemming
        )

        logger.info(f"Text preprocessor initialized: {self.preprocessor.get_stats()}")

        logger.debug(f"Full database path: {self.db_file}")
        logger.debug(f"Database file exists: {os.path.exists(self.db_file)}")

        # Thread-local storage for database connections
        self._local = threading.local()

        # Initialize database schema and BM25 index
        self._init_database_schema()
        self.bm25, self.id_mapping, self.article_count = (
            self._load_or_build_bm25_index()
        )

        logger.info(
            f"Enhanced BM25WikiRM ready with {self.article_count} articles and advanced preprocessing!"
        )

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
        """Initialize SQLite database schema with preprocessing info."""
        db = sqlite3.connect(self.db_file)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                url TEXT,
                content TEXT NOT NULL,
                cleaned_content TEXT,  -- Cleaned Wikipedia content
                tokens TEXT,  -- Space-separated preprocessed tokens for BM25
                raw_tokens TEXT,  -- Space-separated raw tokens for debugging
                preprocessing_version TEXT  -- Track preprocessing version
            )
        """
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_title ON articles(title)")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_preprocessing ON articles(preprocessing_version)"
        )
        db.commit()
        db.close()

    def _get_preprocessing_version(self) -> str:
        """Get current preprocessing version for cache invalidation."""
        return f"v2.0_stem:{self.preprocessor.use_stemming}_lang:{self.preprocessor.language}"

    def _populate_database(self) -> int:
        """Populate database with Wikipedia articles using enhanced preprocessing."""
        logger.info(
            "Populating database with Wikipedia articles and advanced preprocessing..."
        )

        current_version = self._get_preprocessing_version()

        # Check if database needs updating
        db = self._get_db_connection()
        cursor = db.execute(
            "SELECT COUNT(*) FROM articles WHERE preprocessing_version = ?",
            (current_version,),
        )
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            logger.info(
                f"Database already contains {existing_count} articles with current preprocessing"
            )
            return existing_count

        # Check if we need to clear old preprocessing versions
        cursor = db.execute("SELECT COUNT(*) FROM articles")
        total_count = cursor.fetchone()[0]

        if total_count > 0:
            logger.info("Found articles with old preprocessing version, rebuilding...")
            db.execute("DELETE FROM articles")
            db.commit()

        # Load articles and populate database with preprocessing
        articles = WikipediaDataLoader.load_articles(self.max_articles or 1000000)

        batch_size = 1000
        total_inserted = 0

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]

            # Prepare batch data with preprocessing
            batch_data = []
            for article in batch:
                # Clean Wikipedia content
                cleaned_content = self.preprocessor.clean_wikipedia_content(
                    article["text"]
                )

                # Get preprocessed tokens
                processed_tokens = self.preprocessor.preprocess_text(article["text"])
                tokens_str = " ".join(processed_tokens)

                # Get raw tokens for debugging
                raw_tokens = article["text"].lower().split()
                raw_tokens_str = " ".join(raw_tokens)

                batch_data.append(
                    (
                        article["title"],
                        article["url"],
                        article["text"],
                        cleaned_content,
                        tokens_str,
                        raw_tokens_str,
                        current_version,
                    )
                )

            # Insert batch
            db.executemany(
                """INSERT INTO articles
                   (title, url, content, cleaned_content, tokens, raw_tokens, preprocessing_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                batch_data,
            )
            db.commit()

            total_inserted += len(batch)
            if total_inserted % 10000 == 0:
                logger.info(
                    f"Processed and inserted {total_inserted} articles with advanced preprocessing..."
                )

        logger.info(
            f"Database populated with {total_inserted} articles using advanced preprocessing"
        )
        return total_inserted

    def _load_or_build_bm25_index(self):
        """Load existing BM25 index or build a new one with preprocessing."""
        # First, ensure database is populated with current preprocessing
        article_count = self._populate_database()

        current_version = self._get_preprocessing_version()
        versioned_cache = f"{self.bm25_cache}.{current_version}"

        # Try to load cached BM25 index for current preprocessing version
        if os.path.exists(versioned_cache):
            logger.info(
                "Loading cached BM25 index with matching preprocessing version..."
            )
            try:
                with open(versioned_cache, "rb") as f:
                    cached_data = pickle.load(f)

                if isinstance(cached_data, tuple) and len(cached_data) >= 2:
                    bm25, id_mapping = cached_data[:2]
                    logger.info(
                        "BM25 index with advanced preprocessing loaded from cache"
                    )
                    return bm25, id_mapping, article_count

            except Exception as e:
                logger.warning(f"Failed to load BM25 cache: {e}")

        # Build new BM25 index with advanced preprocessing
        logger.info("Building BM25 index with advanced preprocessing...")

        # Load preprocessed tokens from database
        db = self._get_db_connection()
        cursor = db.execute(
            "SELECT id, tokens FROM articles WHERE preprocessing_version = ? ORDER BY id",
            (current_version,),
        )

        corpus = []
        id_mapping = []
        batch_size = 10000

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            for article_id, tokens_str in batch:
                if tokens_str and tokens_str.strip():
                    corpus.append(tokens_str.split())
                    id_mapping.append(article_id)

            if len(corpus) % 50000 == 0:
                logger.info(f"Loaded {len(corpus)} preprocessed documents for BM25...")

        if not corpus:
            raise RuntimeError(
                "No valid preprocessed documents found for BM25 indexing"
            )

        # Build BM25 index
        logger.info(f"Building BM25 index with {len(corpus)} preprocessed documents...")
        bm25 = BM25Okapi(corpus)

        # Cache the index and ID mapping with version
        logger.info("Caching BM25 index with preprocessing version...")
        with open(versioned_cache, "wb") as f:
            pickle.dump((bm25, id_mapping, current_version), f)

        # Clean up old cache files
        try:
            if os.path.exists(self.bm25_cache) and self.bm25_cache != versioned_cache:
                os.remove(self.bm25_cache)
        except:
            pass

        logger.info("BM25 index built and cached with advanced preprocessing!")
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
        STORM-compatible search method using BM25 scoring with advanced preprocessing.
        """
        # Handle STORM calling conventions (same as WikiRM)
        if query is not None:
            query = query
        elif args:
            query = args[0] if len(args) > 0 else None
            if max_results is None and len(args) > 1:
                max_results = args[1]
        elif "query_or_queries" in kwargs and kwargs["query_or_queries"] is not None:
            query = kwargs.pop("query_or_queries", None)
        elif len(args) == 0 and not kwargs:
            return []

        if query is None:
            logger.debug("No valid query provided, returning empty results")
            return []

        # Validate query content - handle both strings and arrays
        if isinstance(query, str):
            if not query.strip():  # Empty string
                logger.debug("Empty string query provided, returning empty results")
                return []
        elif isinstance(query, (list, tuple)):
            if not query or all(
                not str(q).strip() for q in query
            ):  # Empty array or all empty items
                logger.debug(
                    "Empty array query or all empty items provided, returning empty results"
                )
                return []
        else:
            logger.debug(f"Invalid query type {type(query)}, returning empty results")
            return []

        max_results = max_results if max_results is not None else 10
        format_type = format_type if format_type else "rag"

        # Normalize input and filter out None/empty values
        if isinstance(query, str):
            query_list = [query] if query.strip() else []
        else:
            # Filter out None values and empty strings from the list
            query_list = [str(q) for q in query if q is not None and str(q).strip()]

        if not query_list:
            logger.debug("No valid queries after filtering, returning empty results")
            return []

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
        """Search using BM25 scoring with advanced query preprocessing."""
        try:
            # Preprocess query using same pipeline as documents
            query_tokens = self.preprocessor.preprocess_query(query)

            logger.debug(f"Original query: '{query}'")
            logger.debug(f"Preprocessed query tokens: {query_tokens}")

            if not query_tokens:
                logger.debug(
                    "No valid tokens after preprocessing, returning empty results"
                )
                return []

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top results indices
            top_indices = scores.argsort()[-max_results:][::-1]

            # Fetch articles from database
            results = []

            # Use database connection
            db = sqlite3.connect(self.db_file)
            cursor = db.cursor()

            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    # Use ID mapping to get the correct database ID
                    db_id = self.id_mapping[idx]

                    logger.debug(
                        f"Processing BM25 index {idx} -> DB ID {db_id}, score: {scores[idx]:.4f}"
                    )

                    # Fetch article from database
                    cursor.execute(
                        "SELECT title, url, content FROM articles WHERE id = ?",
                        (db_id,),
                    )
                    row = cursor.fetchone()

                    if row:
                        title, url, content = row
                        result_dict = {
                            "content": content,
                            "title": title,
                            "url": url,
                            "source": "enhanced_bm25_wiki",
                            "score": float(scores[idx]),
                        }
                        results.append(result_dict)
                        logger.debug(
                            f"Found relevant result: {title} (score: {scores[idx]:.4f})"
                        )
                    else:
                        logger.warning(
                            f"No article found for BM25 index {idx} (DB ID {db_id})"
                        )

            db.close()

            logger.info(
                f"Query '{query}' returned {len(results)} results with advanced preprocessing"
            )
            return results

        except Exception as e:
            logger.error(f"Enhanced BM25 search failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get statistics about the loaded index and preprocessing."""
        preprocessor_stats = self.preprocessor.get_stats()

        return {
            "num_articles": self.article_count,
            "db_file": self.db_file,
            "bm25_cache": self.bm25_cache,
            "preprocessing_version": self._get_preprocessing_version(),
            "preprocessor_stats": preprocessor_stats,
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
