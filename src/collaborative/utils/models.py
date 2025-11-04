from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, List, Literal, Optional

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
        min_length=3,
        max_length=MAX_HEADING_COUNT,
        description=f"A list of up to {MAX_HEADING_COUNT} main section headings as plain strings (not objects).",
    )

    @field_validator("headings", mode="before")
    @classmethod
    def normalize_headings(cls, v: Any) -> List[str]:
        """
        Normalize headings to plain strings.
        Handles cases where LLM returns [{"title": "..."}, ...] instead of ["...", ...].
        """
        # Accept several input shapes from the LLM and coerce to list of items:
        # - already a list: use as-is
        # - a single string: wrap into [string]
        # - a dict: could be either a wrapper like {"headings": [...]}, or a single heading object like {"title": "..."}
        if isinstance(v, str):
            v = [v]
        elif isinstance(v, dict):
            # Try to find common container keys that hold the actual list
            for candidate in (
                "headings",
                "items",
                "sections",
                "sections_list",
                "titles",
            ):
                if candidate in v and isinstance(v[candidate], list):
                    v = v[candidate]
                    break
                # If container key maps to a dict, treat its keys as headings
                if candidate in v and isinstance(v[candidate], dict):
                    v = list(v[candidate].keys())
                    break
            else:
                # Treat the dict as a single heading object
                v = [v]

        if not isinstance(v, list):
            raise ValueError("headings must be a list")

        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                # Extract from {"title": "..."} or similar structures
                if "title" in item:
                    normalized.append(str(item["title"]))
                elif "heading" in item:
                    normalized.append(str(item["heading"]))
                elif "name" in item:
                    normalized.append(str(item["name"]))
                # If the dict contains a nested list of headings under common keys, extend from it
                elif any(
                    k in item and isinstance(item[k], list)
                    for k in ("headings", "items", "sections", "titles")
                ):
                    for k in ("headings", "items", "sections", "titles"):
                        if k in item and isinstance(item[k], list):
                            for sub in item[k]:
                                if isinstance(sub, str):
                                    normalized.append(sub)
                                elif isinstance(sub, dict):
                                    if "title" in sub:
                                        normalized.append(str(sub["title"]))
                                    elif "heading" in sub:
                                        normalized.append(str(sub["heading"]))
                                    elif "name" in sub:
                                        normalized.append(str(sub["name"]))
                                    elif len(sub) == 1:
                                        normalized.append(str(next(iter(sub.values()))))
                                    else:
                                        # Fallback: use keys if multiple
                                        normalized.extend([str(x) for x in sub.keys()])
                                else:
                                    normalized.append(str(sub))
                            break
                else:
                    # If multiple keys given (e.g., {"Heading A": "...", "Heading B": "..."}),
                    # treat the keys as headings.
                    if len(item) > 1:
                        normalized.extend([str(k) for k in item.keys()])
                    elif len(item) == 1:
                        # Single-key dict: use the sole value
                        normalized.append(str(next(iter(item.values()))))
                    else:
                        raise ValueError(f"Cannot extract heading from dict: {item}")
            else:
                # Try to convert to string
                normalized.append(str(item))

        return normalized


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

    @field_validator("updates", mode="before")
    @classmethod
    def normalize_updates(cls, v: Any) -> List[dict]:
        """
        Normalize updates to list of dicts.
        Handles cases where LLM returns {"id=...": "status", ...} instead of [{"id": "...", "status": "..."}, ...].
        """
        if isinstance(v, list):
            return v
        elif isinstance(v, dict):
            # LLM returned a dict mapping id→status (e.g., {"id=xyz": "pending"})
            # Convert to list of VerifierStatusUpdate-compatible dicts
            normalized = []
            for k, status_val in v.items():
                # Extract the id: strip "id=" prefix if present
                id_str = k.replace("id=", "").strip()
                normalized.append({"id": id_str, "status": status_val})
            return normalized
        else:
            raise ValueError("updates must be a list or dict")
        return v


# endregion Reviewer Feedback Models
