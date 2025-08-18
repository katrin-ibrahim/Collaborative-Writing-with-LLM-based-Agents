from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class SearchResult(BaseModel):
    """Represents a single search result passage."""

    content: str
    source: str
    relevance_score: float = 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "source": self.source,
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
