# src/collaborative/data_models.py
"""
Data models for collaborative writing system.
"""

import operator

from dataclasses import dataclass
from langchain_core.messages import BaseMessage
from typing import Annotated, Any, Dict, List, Optional

from src.utils.data.models import Outline


@dataclass
class ReviewFeedback:
    """Structured feedback from reviewer agent."""

    overall_score: float
    feedback_text: str
    issues_count: int
    recommendations: List[str]


@dataclass
class CollaborationMetrics:
    """Metrics for tracking collaboration progress."""

    iterations: int
    initial_score: float
    final_score: float
    improvement: float
    total_time: float
    convergence_reason: str


@dataclass
class WriterState:
    """State for simplified writer workflow: search → outline → write."""

    messages: Annotated[List[BaseMessage], operator.add]
    topic: str

    # Search phase
    research_queries: List[str]
    search_results: List[Dict[str, Any]]
    organized_knowledge: Optional[Dict[str, Any]]

    # Outline phase
    initial_outline: Optional[Outline]

    # Writing phase
    article_content: str

    # Metadata
    metadata: Dict[str, Any]

    def __init__(self, messages, topic, **kwargs):
        self.messages = messages
        self.topic = topic
        self.research_queries = kwargs.get("research_queries", [])
        self.search_results = kwargs.get("search_results", [])
        self.organized_knowledge = kwargs.get("organized_knowledge")
        self.initial_outline = kwargs.get("initial_outline")
        self.article_content = kwargs.get("article_content", "")
        self.metadata = kwargs.get("metadata", {})
