from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ResearchChunk(BaseModel):
    """Standardized research chunk with consistent fields."""

    chunk_id: str
    description: str
    content: str
    source: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "description": self.description,
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchChunk":
        """Create ResearchChunk from dictionary (e.g., from storage)."""
        # Handle backward compatibility for summary -> description
        description = data.get("description") or data.get("summary", "")

        return cls(
            chunk_id=data.get("chunk_id", ""),
            description=description,
            content=data.get("content", ""),
            source=data.get("source", ""),
            url=data.get("url"),
            metadata=data.get("metadata", {}),
        )

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
            # Generate description from content if not provided
            words = content.split()
            description = " ".join(words[:25]) + ("..." if len(words) > 25 else "")

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "title": self.title,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "heading_count": self.heading_count,
            "headings": self.headings,
            "paragraph_count": self.paragraph_count,
            "analysis_success": self.analysis_success,
        }


class FactCheckResult(BaseModel):
    """Individual fact checking result."""

    claim: str
    sources_found: int
    search_successful: bool
    verified: Optional[bool] = None
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "claim": self.claim,
            "sources_found": self.sources_found,
            "search_successful": self.search_successful,
            "verified": self.verified,
            "search_results": self.search_results,
            "error": self.error,
        }


class SearchResult(BaseModel):
    """Represents a single search result passage."""

    snippets: List[str]
    source: str
    url: Optional[str] = None
    relevance_score: float = 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "snippets": self.snippets,
            "source": self.source,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


class Outline(BaseModel):
    """Hierarchical outline structure."""

    title: str
    headings: List[str]
    subheadings: Dict[str, List[str]] = Field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "headings": self.headings,
            "subheadings": self.subheadings,
        }


class Article(BaseModel):
    """Generated article with metadata."""

    title: str
    content: str
    outline: Optional[Outline] = None
    sections: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "outline": self.outline.to_dict() if self.outline else None,
            "sections": self.sections,
            "metadata": self._serialize_metadata(self.metadata),
        }

    def _serialize_metadata(self, metadata):
        """Serialize metadata to JSON-compatible format."""

        serialized = {}
        for key, value in metadata.items():
            # Skip model objects and other non-serializable types
            if self._is_model_object(value):
                # Store model name/type instead of the object
                if hasattr(value, "__class__"):
                    serialized[key] = f"<{value.__class__.__name__}>"
                else:
                    serialized[key] = "<model_object>"
                continue

            try:
                # Test if value is JSON serializable
                import json

                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable values to strings
                serialized[key] = str(value)
        return serialized

    def _is_model_object(self, obj):
        """Check if object is a model that shouldn't be serialized."""
        import torch

        # Check for PyTorch models
        if hasattr(torch, "nn") and isinstance(obj, torch.nn.Module):
            return True

        # Check for transformers models
        if hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
            if any(
                model_type in class_name
                for model_type in [
                    "ForCausalLM",
                    "Model",
                    "Tokenizer",
                    "Pipeline",
                    "SentenceTransformer",
                    "AutoModel",
                    "AutoTokenizer",
                ]
            ):
                return True

        # Check for large objects (likely models)
        try:
            import sys

            if sys.getsizeof(obj) > 1024 * 1024:  # Objects larger than 1MB
                return True
        except (TypeError, AttributeError):
            pass

        return False

    @classmethod
    def from_dict(cls, data):
        """Create Article from dictionary."""
        outline_data = data.get("outline")
        outline = Outline(**outline_data) if outline_data else None

        return cls(
            title=data["title"],
            content=data["content"],
            outline=outline,
            sections=data.get("sections", {}),
            metadata=data.get("metadata", {}),
        )


class EvaluationResult(BaseModel):
    """Evaluation metrics for a generated article."""

    heading_soft_recall: float
    heading_entity_recall: float
    rouge_1: float
    rouge_l: float
    article_entity_recall: float

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "heading_soft_recall": self.heading_soft_recall,
            "heading_entity_recall": self.heading_entity_recall,
            "rouge_1": self.rouge_1,
            "rouge_l": self.rouge_l,
            "article_entity_recall": self.article_entity_recall,
        }


@dataclass
class ReviewFeedback:
    """Structured feedback from reviewer agent."""

    overall_score: float
    feedback_text: str
    issues_count: int
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "feedback_text": self.feedback_text,
            "issues_count": self.issues_count,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewFeedback":
        """Create from dictionary."""
        return cls(
            overall_score=data["overall_score"],
            feedback_text=data["feedback_text"],
            issues_count=data["issues_count"],
            recommendations=data["recommendations"],
        )


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
class ToolContext:
    """Structured context from tool usage that persists across sections."""

    # Research context
    retrieved_chunks: Dict[str, str] = field(
        default_factory=dict
    )  # chunk_id -> content
    chunk_summaries: Dict[str, str] = field(default_factory=dict)  # chunk_id -> summary

    # Section context
    previous_sections: Dict[str, str] = field(
        default_factory=dict
    )  # section_name -> content
    current_iteration: int = 0

    # Feedback context
    feedback_items: List[Dict[str, Any]] = field(default_factory=list)

    def add_chunks(self, chunks: Dict[str, str]) -> None:
        """Add retrieved chunks to context."""
        self.retrieved_chunks.update(chunks)

    def add_section(self, section_name: str, content: str) -> None:
        """Add completed section to context."""
        self.previous_sections[section_name] = content

    def get_context_summary(self) -> str:
        """Get formatted summary of available context."""
        parts = []

        if self.retrieved_chunks:
            parts.append(f"Retrieved chunks: {len(self.retrieved_chunks)} available")

        if self.previous_sections:
            parts.append(
                f"Previous sections: {', '.join(self.previous_sections.keys())}"
            )

        if self.feedback_items:
            parts.append(f"Feedback items: {len(self.feedback_items)} available")

        return "; ".join(parts) if parts else "No context available"
