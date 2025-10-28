from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Optional

# --- Configuration Values ---

MAX_HEADING_COUNT = 6
# ----------------------------


class SearchSummary(BaseModel):
    source_queries: List[str]
    search_id: str
    results: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ArticleOutlineValidationModel(BaseModel):
    """The title and hierarchical structure for the article."""

    title: str = Field(..., description="The original given title of the article.")
    headings: List[str] = Field(
        ...,
        min_length=3,  # Added min_length for logical outline
        max_length=MAX_HEADING_COUNT,
        description=f"A list of up to {MAX_HEADING_COUNT} main section headings for the article.",
    )


# endregion Writer Validation Models


# region Reviewer Feedback Models


# region Enums
class FeedbackType(str, Enum):
    """Categories of feedback."""

    CITATION_MISSING = "citation_missing"
    CITATION_INVALID = "citation_invalid"
    CLARITY = "clarity"
    ACCURACY = "accuracy"
    STRUCTURE = "structure"
    CONTENT_EXPANSION = "content_expansion"
    REDUNDANCY = "redundancy"
    TONE = "tone"
    DEPTH = "depth"
    FLOW = "flow"


class FeedbackStatus(str, Enum):
    """Feedback lifecycle status."""

    PENDING = "pending"
    ADDRESSED = "addressed"
    WONT_FIX = "wont_fix"
    VERIFIED_ADDRESSED = "verified_addressed"


#  endregion Enums


class FeedbackValidationModel(BaseModel):
    """
    ONE item as emitted by the Reviewer LLM.
    No id, no status, no iteration — Python adds those.
    """

    section: str = Field(description="Target section or '_overall'.")
    type: FeedbackType = Field(description="Feedback category.")
    issue: str = Field(description="What is wrong?")
    suggestion: str = Field(description="How to fix it.")
    priority: str = Field(
        pattern="^(high|medium|low)$", description="Importance: high|medium|low"
    )
    quote: Optional[str] = Field(
        default=None, description="Optional exact excerpt linked to the issue."
    )
    paragraph_number: Optional[int] = Field(
        default=None, description="Optional 1-indexed paragraph."
    )
    location_hint: Optional[str] = Field(
        default=None, description="Optional locator hint within the section."
    )


class ReviewerTaskValidationModel(BaseModel):
    """
    Reviewer LLM output: a flat list of feedback items (+ optional global text).
    """

    items: List[FeedbackValidationModel] = Field(
        description="List of feedback items (no id/status/iteration)."
    )
    overall_assessment: Optional[str] = Field(
        default=None, description="Optional short holistic summary."
    )


# --------- Stored item (single source of truth) ----------
class FeedbackStoredModel(BaseModel):
    """
    Canonical stored feedback item. Mirrors validation fields, plus id/status/writer_comment.
    Iteration is NOT a field — it’s the container key in memory: feedback_by_iteration[iteration][id] = item.
    """

    model_config = ConfigDict(extra="forbid")

    # from validation
    section: str = Field(description="Target section or '_overall'.")
    type: FeedbackType = Field(description="Feedback category.")
    issue: str = Field(description="What is wrong?")
    suggestion: str = Field(description="How to fix it.")
    priority: str = Field(
        pattern="^(high|medium|low)$", description="Importance: high|medium|low"
    )
    quote: Optional[str] = Field(default=None, description="Optional exact excerpt.")
    paragraph_number: Optional[int] = Field(
        default=None, description="Optional 1-indexed paragraph."
    )
    location_hint: Optional[str] = Field(
        default=None, description="Optional locator hint."
    )

    # added by Python
    id: str = Field(description="Globally unique id (e.g., 'intro_iter2_item0').")
    status: FeedbackStatus = Field(
        default=FeedbackStatus.PENDING, description="Current resolution status."
    )
    writer_comment: Optional[str] = Field(
        default=None, description="Writer's note when updating status."
    )


# --------- Writer → validation ----------
class WriterStatusUpdate(BaseModel):
    id: str = Field(description="Existing feedback item id to update.")
    status: FeedbackStatus = Field(description="One of: pending | addressed | wont_fix")
    writer_comment: Optional[str] = Field(
        default=None, description="Short rationale/note."
    )
    # NOTE: if status == VERIFIED_ADDRESSED is returned, we will ignore it (writer can't verify).


class WriterValidationModel(BaseModel):
    """Model for writer's feedback status updates."""

    updates: List[WriterStatusUpdate] = Field(
        description="Per-item status updates from the writer."
    )
    updated_content: str = Field(
        description="The revised content after addressing feedback."
    )
    content_type: Literal["full_article", "partial_section"] = Field(
        description="Indicates if the updated_content is a full article or a partial revision."
    )


class WriterValidationBatchModel(BaseModel):
    """Batch model for multiple writer feedback status updates."""

    items: List[WriterValidationModel] = Field(
        description="List of writer validation models for batch processing."
    )


# --------- Verifier → validation ----------
class VerifierStatusUpdate(BaseModel):
    id: str = Field(description="Existing feedback item id.")
    status: FeedbackStatus = Field(
        description="Use 'verified_addressed' to confirm; 'pending' to revert; may keep 'wont_fix' if justified."
    )


class VerificationValidationModel(BaseModel):
    updates: List[VerifierStatusUpdate] = Field(
        description="Per-item verification decisions."
    )
    summary: str = Field(description="4–6 sentence summary of verification findings.")


# endregion Reviewer Feedback Models
