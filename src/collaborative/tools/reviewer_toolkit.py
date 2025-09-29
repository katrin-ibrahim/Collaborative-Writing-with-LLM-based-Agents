# src/collaborative/tools/reviewer_toolkit.py
"""
Toolkit for ReviewerAgent - exposes verification and analysis tools.
"""

import logging
from typing import Any, Dict, List

from src.collaborative.tools.tool_definitions import (
    get_article_metrics,
    get_chunks_by_ids,
    get_current_iteration,
    get_section_from_iteration,
    verify_claims_with_research,
)

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
        return [
            verify_claims_with_research,  # Primary fact-checking tool (uses existing chunks)
            get_chunks_by_ids,  # Access writer's research chunks for verification
            get_section_from_iteration,  # Compare with previous iterations
            get_current_iteration,  # Understand review context
            get_article_metrics,  # Calculate objective article metrics (word count, structure)
        ]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return descriptions of available tools."""
        return {
            "verify_claims_with_research": """Primary fact-checking tool that verifies claims against writer's research.
            COMPREHENSIVE: Automatically finds relevant chunks and assesses claim accuracy.
            BUILT-IN: Uses existing research chunks from writer - no additional searches needed.""",
            "get_chunks_by_ids": """Access specific research chunks used by writer for detailed verification.
            DEEP-DIVE: Get full content of chunks to verify specific claims or provide detailed feedback.
            WORKFLOW: identify questionable claims → get relevant chunk IDs → retrieve full content for analysis.""",
            "get_section_from_iteration": """Compare section evolution to assess writing improvement quality.
            PROGRESS TRACKING: get_section_from_iteration("Methods", 0) vs iteration 1 to see changes.
            EVALUATION: Use to verify if your previous feedback was properly addressed.""",
            "get_current_iteration": """Understand review context - early drafts need structural feedback, later ones need polish.
            STRATEGY: iteration 0=focus on structure/content, iteration 1+=focus on refinement/accuracy.
            WORKFLOW: Check iteration → adjust review focus and feedback granularity.""",
            "get_article_metrics": """Calculate objective structural metrics for article assessment.
            ANALYSIS: Provides word count, heading count, paragraph count for scope understanding.
            WORKFLOW: Use early in review to understand article structure and adjust feedback depth.""",
        }
