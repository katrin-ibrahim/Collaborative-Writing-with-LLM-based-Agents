from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class SearchResult(BaseModel):
    """Represents a single search result passage."""
    content: str
    source: str
    relevance_score: float = 0.0

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
    rouge_2: float
    rouge_l: float
    article_entity_recall: float