# src/collaborative/data_models.py
"""
Data models for collaborative writing system.
"""

import operator

from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage
from typing import Annotated, Any, Dict, List, Optional


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
class Outline:
    """Article outline structure."""

    title: str
    headings: List[str]
    subheadings: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class WriterState:
    """State for writer workflow with planning-first approach."""

    messages: Annotated[List[BaseMessage], operator.add]
    topic: str

    # Planning phase
    initial_outline: Optional[Outline]
    research_queries: List[str]

    # Research phase
    search_results: List[Dict[str, Any]]
    # Refinement phase
    refined_outline: Optional[Outline]
    knowledge_gaps: List[str]

    # Writing phase
    article_content: str

    # Flow control
    research_iterations: int
    ready_to_write: bool

    # Metadata
    metadata: Dict[str, Any]

    def __init__(self, messages, topic, **kwargs):
        self.messages = messages
        self.topic = topic
        self.initial_outline = kwargs.get("initial_outline")
        self.research_queries = kwargs.get("research_queries", [])
        self.search_results = kwargs.get("search_results", [])
        self.organized_knowledge = kwargs.get("organized_knowledge")
        self.refined_outline = kwargs.get("refined_outline")
        self.knowledge_gaps = kwargs.get("knowledge_gaps", [])
        self.article_content = kwargs.get("article_content", "")
        self.research_iterations = kwargs.get("research_iterations", 0)
        self.ready_to_write = kwargs.get("ready_to_write", False)
        self.metadata = kwargs.get("metadata", {})
