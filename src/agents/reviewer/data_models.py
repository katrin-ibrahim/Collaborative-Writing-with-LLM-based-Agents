# src/agents/reviewer/data_models.py
from enum import Enum

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class FeedbackCategory(Enum):
    """Categories for reviewer feedback."""

    FACTUAL = "factual"
    STRUCTURAL = "structural"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    STYLE = "style"


class Severity(Enum):
    """Severity levels for feedback items."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"


@dataclass
class Claim:
    """Represents a verifiable claim extracted from content."""

    text: str
    section: str
    line_number: Optional[int] = None
    claim_type: str = "factual"  # "factual", "statistical", "historical", etc.
    confidence: float = 0.0  # How confident we are this is a verifiable claim


@dataclass
class FactCheckResult:
    """Result of fact-checking a specific claim."""

    claim: Claim
    accuracy_score: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_status: str  # "verified", "disputed", "unverifiable"
    sources: List[str]


@dataclass
class StructureAnalysis:
    """Analysis of article structure and organization."""

    outline_score: float  # Logical flow of sections
    transition_score: float  # Quality of section transitions
    coherence_score: float  # Overall coherence
    completeness_score: float  # Coverage of topic
    issues: List[str]  # Specific structural problems
    recommendations: List[str]  # Improvement suggestions


@dataclass
class FeedbackItem:
    """Individual feedback item with category and severity."""

    category: FeedbackCategory
    severity: Severity
    description: str
    location: Optional[str] = None  # Section or line reference
    suggestion: str = ""  # Actionable recommendation
    context: Optional[str] = None  # Additional context


@dataclass
class ReviewFeedback:
    """Complete review feedback for an article."""

    overall_score: float
    category_scores: Dict[FeedbackCategory, float]
    issues: List[FeedbackItem]
    recommendations: List[str]
    summary: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "category_scores": {
                cat.value: score for cat, score in self.category_scores.items()
            },
            "issues": [
                {
                    "category": item.category.value,
                    "severity": item.severity.value,
                    "description": item.description,
                    "location": item.location,
                    "suggestion": item.suggestion,
                    "context": item.context,
                }
                for item in self.issues
            ],
            "recommendations": self.recommendations,
            "summary": self.summary,
            "metadata": self.metadata,
        }


class ReviewError(Exception):
    """Base exception for review process errors."""


class ClaimExtractionError(ReviewError):
    """Failed to extract claims from content."""


class FactCheckError(ReviewError):
    """Failed to verify claims against sources."""


class StructureAnalysisError(ReviewError):
    """Failed to analyze article structure."""
