import logging
import re
import wikipediaapi
from typing import List

logger = logging.getLogger(__name__)


class LinkExtractor:
    """Extract structured info for a topic from Wikipedia using page links and categories."""

    def __init__(self, lang: str = "en"):
        # This library is no longer used by extract_topic_data,
        # but is kept for potential future use or consistency.
        self.wiki = wikipediaapi.Wikipedia(language=lang, user_agent="collab-agent/1.0")

    # ----------------- Public Methods -----------------

    def extract_topic_data(self, topic: str) -> List:
        """Return structured data for the topic to feed the query generator."""
        # Use the same 'wikipedia' library as the other methods for consistency
        from wikipedia import DisambiguationError, PageError
        from wikipedia import page as wpage

        try:
            # 1. Get the page object
            topic_page = wpage(
                topic.replace(" ", "_"), auto_suggest=False, redirect=True
            )
        except DisambiguationError:
            logger.warning(f"Topic page is a disambiguation page: {topic}")
            return []
        except PageError:
            logger.warning(f"Topic page does not exist: {topic}")
            return []
        except Exception as e:
            logger.error(f"Error loading topic page '{topic}': {e}")
            return []

        # 2. Get the plain-text summary and the full list of page links
        summary_text = topic_page.summary
        all_link_titles = set(topic_page.links)  # Use a set for fast lookups

        # 3. Cross-reference: Find which links are *mentioned* in the summary
        relevant_links = []
        for title in all_link_titles:
            # We check if the link title (e.g., "Foo Fighters") exists as plain text
            # in the summary.
            # We use \b (word boundary) to ensure we match "Foo Fighters"
            # and not "Foo Fighters-related".
            if re.search(r"\b" + re.escape(title) + r"\b", summary_text, re.IGNORECASE):

                # 4. Apply our relevance filter to the matches
                if self._is_relevant_link(title):
                    relevant_links.append(title)

        return relevant_links

    # ----------------- Filtering -----------------

    @staticmethod
    def _is_relevant_link(title: str) -> bool:
        """Filters out links that are likely not useful RAG context."""
        lower = title.lower()

        # Filter "List of..." pages
        if lower.startswith("list of"):
            return False

        # Filter disambiguation pages
        if "(disambiguation)" in lower:
            return False

        # Filter out broad year/date/century pages
        if re.fullmatch(r"^\d{1,4}$", lower):  # "1972", "2022"
            return False
        if re.fullmatch(r"^\d{1,4}s$", lower):  # "1970s", "2020s"
            return False
        if " century" in lower:  # "20th century", "21st-century"
            return False

        # Filter out "CS1" and other meta-templates
        if lower.startswith("cs1 ") or lower.startswith("articles with"):
            return False

        return True
