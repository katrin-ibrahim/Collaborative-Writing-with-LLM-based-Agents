# src/collaborative/tools/reviewer_toolkit.py
"""
Toolkit for ReviewerAgent - exposes verification and analysis tools.
"""

import logging
from typing import Any, Dict, List

from src.collaborative.tools.tool_definitions import search_and_retrieve

logger = logging.getLogger(__name__)


class ReviewerToolkit:
    """
    Toolkit providing tools for ReviewerAgent.

    Exposes tools relevant for fact-checking and content analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_available_tools(self) -> List:
        """Return list of available tools for the reviewer agent."""
        return [search_and_retrieve]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return descriptions of available tools."""
        return {
            "search_and_retrieve": "Search external sources for fact-checking and verification",
            "get_article_metrics": "Get objective metrics about article structure, such as word count, heading count, and paragraph count",
        }
