import logging
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResearchChunk(BaseModel):
    """Standardized research chunk with consistent fields."""

    chunk_id: str
    description: str
    content: str
    source: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # --- new optional retrieval metadata ---
    relevance_score_normalized: Optional[float] = (
        None  # normalized 0..1, faiss already normalized
    )
    rank: Optional[int] = None  # position in ranked list

    @classmethod
    def from_retrieval_result(
        cls, chunk_id: str, result: Dict[str, Any]
    ) -> "ResearchChunk":
        """
        Create ResearchChunk from retrieval manager result.
        Handles different retrieval result formats.
        """
        # Extract content from various possible fields
        snippets = result.get("snippets")
        if isinstance(snippets, list) and snippets:
            # Join multiple snippets or take the first one
            content = " ".join(snippets) if len(snippets) > 1 else snippets[0]
        elif isinstance(snippets, str):
            content = snippets
        else:
            # Fallback to other possible content fields
            content = result.get("content", "") or ""

        # Extract description from various possible fields (prefer summary/description over content)
        description = result.get("description")

        if not description and content:
            logger.error(
                f"Missing description in retrieval result for chunk_id {chunk_id}"
            )
            description = ""

        # Ensure description is always a string
        if description is None:
            description = ""

        return cls(
            chunk_id=chunk_id,
            description=description,
            content=content,
            source=result.get("source", ""),
            url=result.get("url"),
            metadata=result.get("metadata", {}),
        )


class ArticleMetrics(BaseModel):
    """Structured metrics for article analysis."""

    title: str
    word_count: int
    character_count: int
    heading_count: int
    headings: List[str]
    paragraph_count: int
    analysis_success: bool = True


class FactCheckResult(BaseModel):
    """Individual fact checking result."""

    claim: str
    sources_found: int
    search_successful: bool
    verified: Optional[bool] = None
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class Outline(BaseModel):
    """Hierarchical outline structure."""

    title: str
    headings: List[str]
    subheadings: Dict[str, List[str]] = Field(default_factory=dict)


class Article(BaseModel):
    """Generated article with metadata."""

    title: str
    content: str
    outline: Optional[Outline] = None
    sections: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Evaluation metrics for a generated article."""

    heading_soft_recall: float
    heading_entity_recall: float
    rouge_1: float
    rouge_l: float
    article_entity_recall: float


@dataclass
class ReviewFeedback(BaseModel):
    """Structured feedback from reviewer agent."""

    overall_score: float
    feedback_text: str
    issues_count: int
    recommendations: List[str]
