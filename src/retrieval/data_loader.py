"""
Shared Wikipedia Data Loader for Retrieval Managers
Loads Wikipedia articles from local XML dumps to eliminate HuggingFace streaming complexity.
"""

import glob

import json
import logging
import os
import pickle
from typing import Dict, List

logger = logging.getLogger(__name__)


class WikipediaDataLoader:
    """Loads Wikipedia data from local XML dumps for consistent comparison."""

    @staticmethod
    def load_wikipedia_dump(dump_dir: str) -> List[Dict]:
        """
        Load Wikipedia articles from local WikiExtractor JSON dump.

        Args:
            dump_dir: Directory containing WikiExtractor output (wiki_*.json files)

        Returns:
            List of dicts with 'title', 'text', 'url' keys
        """
        logger.info(f"Loading Wikipedia articles from dump directory: {dump_dir}")

        articles = []
        json_files = glob.glob(os.path.join(dump_dir, "**/wiki_*.json"), recursive=True)

        if not json_files:
            raise FileNotFoundError(f"No wiki_*.json files found in {dump_dir}")

        logger.info(f"Found {len(json_files)} dump files to process")

        for file_path in json_files:
            logger.debug(f"Processing {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            article_data = json.loads(line)

                            # Filter out very short articles
                            if len(article_data.get("text", "")) > 500:
                                articles.append(
                                    {
                                        "title": article_data["title"],
                                        "text": article_data["text"],
                                        "url": article_data.get(
                                            "url",
                                            f"https://en.wikipedia.org/wiki/{article_data['title'].replace(' ', '_')}",
                                        ),
                                    }
                                )
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"JSON decode error in {file_path} line {line_num}: {e}"
                            )
                            continue

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(articles)} articles from Wikipedia dump")
        return articles

    @staticmethod
    def load_articles(
        num_articles: int = 100000,
        cache_file: str = "wikipedia_articles.pkl",
        dump_dir: str = None,
    ) -> List[Dict]:
        """
        Load Wikipedia articles from local dump with caching.

        Args:
            num_articles: Number of articles to load (for testing)
            cache_file: Cache file to avoid reloading
            dump_dir: Directory containing WikiExtractor output

        Returns:
            List of dicts with 'title', 'text', 'url' keys
        """
        if os.path.exists(cache_file):
            logger.info(f"Loading cached Wikipedia articles from {cache_file}")
            with open(cache_file, "rb") as f:
                cached_articles = pickle.load(f)
                # Return requested number of articles
                return cached_articles[:num_articles]

        # Determine dump directory
        if dump_dir is None:
            # Default locations to check
            possible_dirs = [
                "data/wiki_dump/text",
                "../data/wiki_dump/text",
            ]

            dump_dir = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path) and glob.glob(
                    os.path.join(dir_path, "**/wiki_*.json"), recursive=True
                ):
                    dump_dir = dir_path
                    break

            if dump_dir is None:
                raise FileNotFoundError(
                    "No Wikipedia dump found. Please run:\n ./scripts/setup_wikipedia_dump.sh"
                )

        logger.info(f"Loading Wikipedia articles from dump: {dump_dir}")

        # Load all articles from dump
        all_articles = WikipediaDataLoader.load_wikipedia_dump(dump_dir)

        # Cache the full dataset for future use
        logger.info(f"Caching {len(all_articles)} articles to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(all_articles, f)

        # Return requested number of articles
        return all_articles[:num_articles]
