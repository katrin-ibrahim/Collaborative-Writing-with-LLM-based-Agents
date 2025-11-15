import logging
import wikipediaapi
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class CategoryExtractor:
    """Extract Wikipedia category members to find related articles."""

    def __init__(self, lang: str = "en", user_agent: str = "collab-agent/1.0"):
        self.wiki = wikipediaapi.Wikipedia(language=lang, user_agent=user_agent)

    def get_topic_categories(self, topic: str) -> List[str]:
        """Get categories that the topic page belongs to, filtered for relevance."""
        # Normalize topic name
        normalized_topic = topic.replace("_", " ")

        page = self.wiki.page(normalized_topic)
        if not page.exists():
            logger.warning(f"Topic page does not exist: {normalized_topic}")
            return []

        # Extract category names (without "Category:" prefix)
        categories = []
        for cat_title in page.categories.keys():
            # Remove "Category:" prefix
            cat_name = cat_title.replace("Category:", "")

            # Filter out meta/maintenance categories
            if self._is_relevant_category(cat_name):
                categories.append(cat_name)

        logger.info(
            f"Found {len(categories)} relevant categories (filtered from {len(page.categories)}) for topic: {normalized_topic}"
        )
        return categories

    @staticmethod
    def _is_relevant_category(cat_name: str) -> bool:
        """Filter out Wikipedia maintenance and meta categories."""
        lower = cat_name.lower()

        # Filter meta/maintenance categories
        meta_keywords = [
            "all wikipedia articles",
            "articles with",
            "articles using",
            "articles lacking",
            "articles needing",
            "accuracy disputes",
            "all articles",
            "cs1 ",
            "pages using",
            "pages with",
            "wikipedia articles",
            "webarchive template",
            "official website",
            "short description",
            "wikidata",
            "commons category",
        ]

        for keyword in meta_keywords:
            if keyword in lower:
                return False

        return True

    def get_category_members(
        self,
        category_name: str,
        max_members: int = 15,
        exclude_topics: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Get article titles from a Wikipedia category.

        Args:
            category_name: Name of category (without "Category:" prefix)
            max_members: Maximum number of members to return
            exclude_topics: Set of topic titles to exclude (e.g., the main topic itself)

        Returns:
            List of article titles from the category
        """
        exclude_topics = exclude_topics or set()

        # Add "Category:" prefix for API
        cat_page = self.wiki.page(f"Category:{category_name}")

        if not cat_page.exists():
            logger.warning(f"Category does not exist: {category_name}")
            return []

        members = []
        for title, member in cat_page.categorymembers.items():
            # Only include articles (namespace 0), not subcategories or files
            if member.ns == 0:
                # Exclude the main topic itself
                if title not in exclude_topics:
                    members.append(title)

                    if len(members) >= max_members:
                        break

        logger.info(
            f"Extracted {len(members)} article members from category: {category_name}"
        )
        return members

    def get_related_articles(
        self, topic: str, max_categories: int = 3, max_members_per_category: int = 5
    ) -> List[str]:
        """
        Get related articles by extracting members from topic's categories.

        Args:
            topic: Topic title
            max_categories: Maximum number of categories to process
            max_members_per_category: Maximum members to extract from each category

        Returns:
            List of related article titles (deduplicated)
        """
        # Get topic's categories
        categories = self.get_topic_categories(topic)

        if not categories:
            return []

        # Limit categories to process
        categories = categories[:max_categories]

        # Create exclusion set with the topic itself
        exclude = {topic, topic.replace("_", " ")}

        # Collect members from each category
        all_members = set()
        for cat_name in categories:
            members = self.get_category_members(
                cat_name, max_members=max_members_per_category, exclude_topics=exclude
            )
            all_members.update(members)

        result = list(all_members)
        logger.info(
            f"Found {len(result)} unique related articles from {len(categories)} categories"
        )
        return result
