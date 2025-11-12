"""
Wikipedia category fetcher using the `wikipedia-api` pip package.

This implementation uses `wikipediaapi` to retrieve the page's categories
as metadata (no article content is downloaded beyond what the library fetches
from the MediaWiki API). Categories are public metadata and are safe to use
for informing retrieval queries (not data leakage of FreshWiki content).
"""

import logging
import wikipediaapi
from typing import List

logger = logging.getLogger(__name__)


class WikipediaCategoryFetcher:
    """Fetches Wikipedia categories for topics using wikipediaapi."""

    def __init__(self, lang: str = "en"):
        """Initialize the wikipediaapi client.

        Args:
            lang: Language code for Wikipedia (default 'en')
        """
        # wikipediaapi optionally accepts a user_agent kwarg; provide one
        # to satisfy stricter type-checkers and polite API usage.
        self.wiki = wikipediaapi.Wikipedia(language=lang, user_agent="collab-agent/1.0")
        logger.info(f"Initialized wikipediaapi client for lang={lang}")

    def get_categories(self, topic: str, max_categories: int = 10) -> List[str]:
        """Return a short list of category names for the given topic.

        Args:
            topic: Topic title (e.g., "2022 AFL Grand Final")
            max_categories: Maximum number of categories to return

        Returns:
            List of category names (without the "Category:" prefix)
        """
        page_title = topic.replace(" ", "_")
        page = self.wiki.page(page_title)

        if not page.exists():
            logger.warning(f"Wikipedia page does not exist for: {topic} ({page_title})")
            return []

        # wikipediaapi returns a dict-like mapping of category title->Category
        raw_categories = page.categories or {}

        categories: List[str] = []
        for full_title in raw_categories.keys():
            # full_title is like 'Category:2022 AFL season'
            name = full_title.replace("Category:", "")
            if not _is_maintenance_category(name):
                categories.append(name)
            if len(categories) >= max_categories:
                break

        logger.info(f"Found {len(categories)} categories for '{topic}'")
        return categories


def _is_maintenance_category(name: str) -> bool:
    """Heuristic filter to remove maintenance/meta categories."""
    lower = name.lower()
    patterns = [
        "all articles",
        "articles with",
        "pages with",
        "pages using",
        "cs1",
        "webarchive",
        "wikipedia articles",
        "use dmy dates",
        "use mdy dates",
        "articles needing",
        "articles containing",
        "good articles",
        "featured articles",
        "commons category",
    ]
    return any(lower.startswith(p) for p in patterns)


def _test():
    fetcher = WikipediaCategoryFetcher()
    topics = [
        "2022 AFL Grand Final",
        "Taylor Hawkins",
        "Hurricane Hilary",
        "2022 Crimean Bridge explosion",
    ]

    for t in topics:
        print(f"\nTopic: {t}")
        cats = fetcher.get_categories(t, max_categories=8)
        if cats:
            for i, c in enumerate(cats, 1):
                print(f"  {i}. {c}")
        else:
            print("  (no categories)")


if __name__ == "__main__":
    _test()
