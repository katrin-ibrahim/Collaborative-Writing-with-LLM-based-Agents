# src/collaborative/tools/writer_toolkit.py
"""
Toolkit for WriterAgent - exposes research tools.
"""

from typing import Dict, List

from src.collaborative.tools.tool_definitions import search_and_retrieve


class WriterToolkit:
    """
    Toolkit providing tools for WriterAgent.

    Exposes only tools relevant for writing and research.
    """

    def __init__(self, config: Dict[str, any]):
        self.config = config

    def get_available_tools(self) -> List:
        """Return list of available tools for the writer agent."""
        return [search_and_retrieve]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return descriptions of available tools."""
        return {
            "search_and_retrieve": "Search external sources for information and content research"
        }
