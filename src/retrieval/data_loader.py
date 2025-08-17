"""
Enhanced Wikipedia Data Loader with Robust JSON Parsing
Handles corrupted WikiExtractor output with graceful degradation and recovery.
"""

import glob

import json
import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WikipediaDataLoader:
    """Enhanced Wikipedia data loader with robust JSON parsing and corruption recovery."""

    @staticmethod
    def load_wikipedia_dump(dump_dir: str, max_articles: int = None) -> List[Dict]:
        """
        Load Wikipedia articles from local WikiExtractor JSON dump with robust error handling.

        Args:
            dump_dir: Directory containing WikiExtractor output (wiki_*.json files)
            max_articles: Optional limit on number of articles to load (for efficiency)

        Returns:
            List of dicts with 'title', 'text', 'url' keys
        """
        logger.info(f"Loading Wikipedia articles from dump directory: {dump_dir}")

        articles = []
        corruption_stats = {
            "total_files": 0,
            "corrupted_files": 0,
            "total_lines": 0,
            "corrupted_lines": 0,
            "recovered_articles": 0,
            "skipped_articles": 0,
        }

        json_files = glob.glob(os.path.join(dump_dir, "**/wiki_*"), recursive=True)

        if not json_files:
            raise FileNotFoundError(f"No wiki_* files found in {dump_dir}")

        logger.info(f"Found {len(json_files)} dump files to process")
        corruption_stats["total_files"] = len(json_files)

        for file_path in json_files:
            logger.debug(f"Processing {file_path}")
            file_articles, file_stats = WikipediaDataLoader._process_file_robust(
                file_path
            )
            articles.extend(file_articles)

            # Update corruption statistics
            corruption_stats["total_lines"] += file_stats["total_lines"]
            corruption_stats["corrupted_lines"] += file_stats["corrupted_lines"]
            corruption_stats["recovered_articles"] += file_stats["recovered_articles"]
            corruption_stats["skipped_articles"] += file_stats["skipped_articles"]

            if file_stats["corrupted_lines"] > 0:
                corruption_stats["corrupted_files"] += 1

            # Early termination if we have enough articles
            if max_articles and len(articles) >= max_articles:
                logger.info(
                    f"Reached target of {max_articles} articles, stopping early"
                )
                articles = articles[:max_articles]  # Truncate to exact number
                break

        # Report corruption statistics
        WikipediaDataLoader._report_corruption_stats(corruption_stats)

        logger.info(f"Successfully loaded {len(articles)} articles from Wikipedia dump")
        return articles

    @staticmethod
    def _process_file_robust(file_path: str) -> Tuple[List[Dict], Dict]:
        """
        Process a single file with robust error handling and corruption recovery.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple of (articles_list, statistics_dict)
        """
        articles = []
        stats = {
            "total_lines": 0,
            "corrupted_lines": 0,
            "recovered_articles": 0,
            "skipped_articles": 0,
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if entire file is corrupted
            if not content.strip():
                logger.warning(f"Empty file: {file_path}")
                return articles, stats

            # Try to process line by line first
            lines = content.split("\n")
            stats["total_lines"] = len([line for line in lines if line.strip()])

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Try standard JSON parsing first
                article = WikipediaDataLoader._parse_json_line(
                    line, file_path, line_num
                )

                if article is None:
                    # Try corruption recovery
                    article = WikipediaDataLoader._recover_corrupted_json(
                        line, file_path, line_num
                    )
                    if article:
                        stats["recovered_articles"] += 1
                    else:
                        stats["corrupted_lines"] += 1
                        stats["skipped_articles"] += 1
                        continue

                # Validate and filter article
                if WikipediaDataLoader._is_valid_article(article):
                    articles.append(article)
                else:
                    stats["skipped_articles"] += 1

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            stats["corrupted_lines"] = stats["total_lines"]

        return articles, stats

    @staticmethod
    def _parse_json_line(line: str, file_path: str, line_num: int) -> Optional[Dict]:
        """
        Parse a single JSON line with standard method.

        Args:
            line: JSON line to parse
            file_path: File path for logging
            line_num: Line number for logging

        Returns:
            Parsed article dict or None if parsing fails
        """
        try:
            article_data = json.loads(line)

            # Ensure required fields exist
            if not all(key in article_data for key in ["title", "text"]):
                logger.warning(
                    f"Missing required fields in {file_path} line {line_num}"
                )
                return None

            return {
                "title": article_data["title"],
                "text": article_data["text"],
                "url": article_data.get(
                    "url",
                    f"https://en.wikipedia.org/wiki/{article_data['title'].replace(' ', '_')}",
                ),
            }

        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.warning(f"Unexpected error parsing {file_path} line {line_num}: {e}")
            return None

    @staticmethod
    def _recover_corrupted_json(
        line: str, file_path: str, line_num: int
    ) -> Optional[Dict]:
        """
        Attempt to recover data from corrupted JSON lines.

        Args:
            line: Corrupted JSON line
            file_path: File path for logging
            line_num: Line number for logging

        Returns:
            Recovered article dict or None if recovery fails
        """
        logger.debug(f"Attempting corruption recovery for {file_path} line {line_num}")

        try:
            # Strategy 1: Try to fix common JSON issues
            fixed_line = WikipediaDataLoader._fix_common_json_issues(line)
            if fixed_line != line:
                try:
                    article_data = json.loads(fixed_line)
                    return WikipediaDataLoader._extract_article_data(article_data)
                except json.JSONDecodeError:
                    pass

            # Strategy 2: Extract data with regex if JSON structure is broken
            article_data = WikipediaDataLoader._extract_with_regex(line)
            if article_data:
                return article_data

            # Strategy 3: Try partial JSON parsing
            article_data = WikipediaDataLoader._parse_partial_json(line)
            if article_data:
                return article_data

        except Exception as e:
            logger.debug(f"Recovery failed for {file_path} line {line_num}: {e}")

        return None

    @staticmethod
    def _fix_common_json_issues(line: str) -> str:
        """
        Fix common JSON formatting issues.

        Args:
            line: Potentially corrupted JSON line

        Returns:
            Fixed JSON line
        """
        fixed = line

        # Fix missing closing braces/brackets
        open_braces = fixed.count("{") - fixed.count("}")
        if open_braces > 0:
            fixed += "}" * open_braces

        open_brackets = fixed.count("[") - fixed.count("]")
        if open_brackets > 0:
            fixed += "]" * open_brackets

        # Fix unclosed quotes in common patterns
        if fixed.count('"') % 2 == 1:
            # Try to close the last quote if it looks like a text field
            if '"text"' in fixed and fixed.endswith('"'):
                pass  # Already ends with quote
            elif '"text":' in fixed:
                fixed += '"'

        return fixed

    @staticmethod
    def _extract_with_regex(line: str) -> Optional[Dict]:
        """
        Extract article data using regex patterns when JSON parsing fails.

        Args:
            line: Line to extract from

        Returns:
            Extracted article dict or None
        """
        try:
            # Extract title
            title_match = re.search(r'"title":\s*"([^"]*)"', line)
            if not title_match:
                return None
            title = title_match.group(1)

            # Extract text (more complex due to potential quotes in content)
            text_match = re.search(
                r'"text":\s*"(.*?)(?:",\s*"|\",\s*$|"$)', line, re.DOTALL
            )
            if not text_match:
                # Try alternative pattern for text at end of line
                text_match = re.search(r'"text":\s*"(.*)', line, re.DOTALL)
                if text_match:
                    text = text_match.group(1)
                    # Remove trailing characters that aren't part of text
                    text = re.sub(r'["}]*$', "", text)
                else:
                    return None
            else:
                text = text_match.group(1)

            # Extract URL if available
            url_match = re.search(r'"url":\s*"([^"]*)"', line)
            url = (
                url_match.group(1)
                if url_match
                else f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            )

            # Basic validation
            if len(title) < 1 or len(text) < 100:
                return None

            return {
                "title": title,
                "text": text,
                "url": url,
            }

        except Exception:
            return None

    @staticmethod
    def _parse_partial_json(line: str) -> Optional[Dict]:
        """
        Try to parse partial JSON by truncating at various points.

        Args:
            line: Line to parse

        Returns:
            Parsed article dict or None
        """
        # Find potential JSON end points
        potential_ends = []

        # Look for complete field patterns
        for match in re.finditer(r'"[^"]+"\s*:\s*"[^"]*"', line):
            potential_ends.append(match.end())

        # Try parsing truncated versions
        for end_pos in sorted(potential_ends, reverse=True):
            try:
                truncated = line[:end_pos] + "}"
                if truncated.startswith("{"):
                    article_data = json.loads(truncated)
                    return WikipediaDataLoader._extract_article_data(article_data)
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _extract_article_data(article_data: Dict) -> Optional[Dict]:
        """
        Extract and validate article data from parsed JSON.

        Args:
            article_data: Parsed JSON data

        Returns:
            Clean article dict or None
        """
        try:
            if not isinstance(article_data, dict):
                return None

            title = article_data.get("title", "").strip()
            text = article_data.get("text", "").strip()

            if not title or not text:
                return None

            return {
                "title": title,
                "text": text,
                "url": article_data.get(
                    "url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                ),
            }

        except Exception:
            return None

    @staticmethod
    def _is_valid_article(article: Dict) -> bool:
        """
        Validate article meets minimum quality requirements.

        Args:
            article: Article dictionary

        Returns:
            True if article is valid
        """
        try:
            # Check required fields
            if not all(key in article for key in ["title", "text", "url"]):
                return False

            # Check minimum content length
            if len(article["text"]) < 500:
                return False

            # Check title is reasonable
            if len(article["title"]) < 2 or len(article["title"]) > 200:
                return False

            # Check for obvious corruption patterns
            text = article["text"]
            if text.count('"') > len(text) * 0.1:  # Too many quotes suggest corruption
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def _report_corruption_stats(stats: Dict):
        """
        Report corruption statistics to help with debugging.

        Args:
            stats: Statistics dictionary
        """
        total_lines = stats["total_lines"]
        corrupted_lines = stats["corrupted_lines"]
        recovered = stats["recovered_articles"]

        if corrupted_lines > 0:
            corruption_rate = (
                (corrupted_lines / total_lines) * 100 if total_lines > 0 else 0
            )
            recovery_rate = (
                (recovered / corrupted_lines) * 100 if corrupted_lines > 0 else 0
            )

            logger.warning(
                f"Data corruption detected in {stats['corrupted_files']}/{stats['total_files']} files"
            )
            logger.warning(
                f"Corruption rate: {corruption_rate:.1f}% ({corrupted_lines}/{total_lines} lines)"
            )
            logger.warning(
                f"Recovery rate: {recovery_rate:.1f}% ({recovered}/{corrupted_lines} lines recovered)"
            )
            logger.warning(f"Total articles skipped: {stats['skipped_articles']}")

            if corruption_rate > 10:
                logger.error(
                    "High corruption rate detected! Consider re-running WikiExtractor"
                )
        else:
            logger.info("No corruption detected in dump files")

    @staticmethod
    def load_articles(
        num_articles: int = 100000,
        cache_file: str = "wikipedia_articles.pkl",
        dump_dir: str = None,
    ) -> List[Dict]:
        """
        Load Wikipedia articles from local dump with caching and robust error handling.

        Args:
            num_articles: Number of articles to load (for testing)
            cache_file: Cache file to avoid reloading
            dump_dir: Directory containing WikiExtractor output

        Returns:
            List of dicts with 'title', 'text', 'url' keys
        """
        # Use project root directory for cache file, not module directory
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        cache_file = os.path.join(project_root, cache_file)
        if os.path.exists(cache_file):
            logger.info(f"Loading cached Wikipedia articles from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached_articles = pickle.load(f)
                    return cached_articles[:num_articles]
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
                logger.info("Proceeding with fresh dump processing")

        # Determine dump directory
        if dump_dir is None:
            possible_dirs = [
                "data/wiki_dump/text",
                "../data/wiki_dump/text",
            ]

            dump_dir = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path) and glob.glob(
                    os.path.join(dir_path, "**/wiki_*"), recursive=True
                ):
                    dump_dir = dir_path
                    break

            if dump_dir is None:
                raise FileNotFoundError(
                    "No Wikipedia dump found. Please run:\n ./scripts/setup_wikipedia_dump.sh"
                )

        logger.info(f"Loading Wikipedia articles from dump: {dump_dir}")

        # Load articles from dump with robust parsing (limit to requested number)
        all_articles = WikipediaDataLoader.load_wikipedia_dump(
            dump_dir, max_articles=num_articles
        )

        if len(all_articles) == 0:
            raise RuntimeError(
                "No valid articles found in dump. Check WikiExtractor output."
            )

        # Cache the full dataset for future use
        try:
            logger.info(f"Caching {len(all_articles)} articles to {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(all_articles, f)
        except Exception as e:
            logger.warning(f"Failed to cache articles: {e}")

        # Return requested number of articles
        return all_articles[:num_articles]
