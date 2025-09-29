# src/collaborative/tools/writer_toolkit.py
"""
Toolkit for WriterAgent - exposes research and memory tools.
"""

from typing import Dict, List

from src.collaborative.tools.tool_definitions import (
    get_chunks_by_ids,
    get_current_iteration,
    get_feedback,
    get_section_from_iteration,
    search_and_retrieve,
)


class WriterToolkit:
    """
    Toolkit providing tools for WriterAgent.

    Exposes only tools relevant for writing and research.
    """

    def __init__(self, config: Dict[str, any]):
        self.config = config

    def get_available_tools(self) -> List:
        """Return list of available tools for the writer agent."""
        return [
            search_and_retrieve,  # Research external sources
            get_chunks_by_ids,  # Retrieve research content for writing
            get_section_from_iteration,  # Compare with previous section versions
            get_current_iteration,  # Understand current revision stage
            get_feedback,  # Access reviewer feedback for improvements
        ]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return comprehensive tool descriptions with usage patterns for the writer agent."""
        return {
            "search_and_retrieve": """Research external sources for writing content. Returns chunk summaries with IDs.
            WORKFLOW: search_and_retrieve(query) → use returned chunk_ids with get_chunks_by_ids() for full content.
            CHAINING: Multiple searches for different aspects, then retrieve specific chunks for writing.""",
            "get_chunks_by_ids": """Retrieve full content from research chunks for writing. Use after search_and_retrieve.
            USAGE: get_chunks_by_ids("chunk_id1,chunk_id2") - comma-separated IDs from search results.
            PATTERNS: Get chunks in batches, integrate content into your writing sections.""",
            "get_section_from_iteration": """Access previous section versions to track your writing evolution.
            COMPARISON: Call twice with different iterations to see changes you made.
            WORKFLOW: get_section_from_iteration("Intro", 0) vs get_section_from_iteration("Intro", 1)""",
            "get_current_iteration": """Know which writing iteration you're in for context-aware decisions.
            USAGE: iteration 0=initial draft, 1+=revisions. Affects strategy: early=structure, later=polish.
            WORKFLOW: Check iteration → adjust writing approach accordingly.""",
            "get_feedback": """Access reviewer feedback to guide improvements in subsequent iterations.
            IMPROVEMENT: Review previous feedback to understand what needs to be addressed.
            WORKFLOW: get_feedback() → analyze issues → make targeted improvements → write better content.""",
        }
